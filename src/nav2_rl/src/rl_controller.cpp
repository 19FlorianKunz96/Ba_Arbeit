#include "nav2_rl/rl_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <algorithm>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // doTransform
using namespace std::chrono_literals;

namespace nav2_rl {

void RLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;

  agent_service_     = node_->declare_parameter<std::string>(name_ + ".agent_service", "/agents/dqn/get_action");
  timeout_ms_        = node_->declare_parameter<int>(name_ + ".timeout_ms", 150); //so lange wird auf eine Antwort des Servers gewartet
  max_lin_           = node_->declare_parameter<double>(name_ + ".max_lin_vel", 0.22);
  max_ang_           = node_->declare_parameter<double>(name_ + ".max_ang_vel", 1.5);
  lookahead_dist_    = node_->declare_parameter<double>(name_ + ".lookahead_dist", 1.0); //wie weit soll der lokale zielpunkt entlang des globalen pfades weg sein
  reach_radius_      = node_->declare_parameter<double>(name_ + ".reach_radius", 0.20); //Radius um Ziel, in dem es als erreicht gilt und neues lokales Goal gepublisht wird
  advance_hysteresis_= node_->declare_parameter<double>(name_ + ".advance_hysteresis", 0.05); //Mindestverschiebung, um ein neues lokales Ziel zu publizieren 
  //(damit nicht bei kleinsten Änderungen neu gesendet wird). Ist das sinnvoll ? Dann wird ja wenn das Ziel weniger als das weg liegt, nichts mehr gepublisht???

  // Publisher für lokales Ziel
  rclcpp::QoS qos(1); qos.transient_local().reliable();
  local_goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("/rl/local_goal", qos);
  global_path_pub = node_->create_publisher<nav_msgs::msg::Path>("/rl/target_path", qos);

  connectClient_();

  // Param-Callback (Handle speichern!)
  param_cb_handle_ = node_->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter>& params){
      for (auto & p : params) {
        if (p.get_name() == name_ + ".agent_service") {
          agent_service_ = p.as_string();
          connectClient_();
        }
      }
      rcl_interfaces::msg::SetParametersResult r; r.successful = true; return r;
    });
}

void RLController::connectClient_() {
  client_ = node_->create_client<turtlebot3_msgs::srv::RLLocalPath>(agent_service_);
}

//liefert die geschätzte Pose des Roboters im map-Frame. Wenn das TF nicht klappt, bekommst du eine Pose im Base-Frame als Notlösung.
geometry_msgs::msg::PoseStamped RLController::robotPoseInMap_() {
  geometry_msgs::msg::PoseStamped base, map;
  base.header.frame_id = costmap_ros_->getBaseFrameID();
  base.header.stamp = node_->now();
  base.pose.orientation.w = 1.0;
  try {
    auto tf_msg = tf_->lookupTransform("map", base.header.frame_id, tf2::TimePointZero);
    tf2::doTransform(base, map, tf_msg);
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "TF lookup failed: %s", e.what());
    map = base; // Fallback
  }
  return map;
}

//euklidische Distanz zwischen zwei Punkten
double RLController::dist2_(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

//Findet den Index der Pose im Pfad, die am nächsten zum gegebenen Punkt p liegt (z.B. der Roboterposition).
int RLController::nearestIndexTo_(const std::vector<geometry_msgs::msg::PoseStamped> & poses,
                                  const geometry_msgs::msg::Point & p) {
  int best = 0; double best_d = 1e9;
  for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
    double d = dist2_(poses[i].pose.position, p);
    if (d < best_d) { best_d = d; best = i; }
  }
  return best;
}

//Gibt den Index auf dem Pfad zurück, der in etwa L Meter vor start_idx liegt, ansonsten das Pfadende.
int RLController::indexAtLookaheadFrom_(const std::vector<geometry_msgs::msg::PoseStamped> & poses,int start_idx, double L)
{
  if (poses.empty()) return 0;
  double acc = 0.0;
  for (int i = start_idx; i < static_cast<int>(poses.size()) - 1; ++i) {
    acc += dist2_(poses[i+1].pose.position, poses[i].pose.position);
    if (acc >= L) return i + 1;
  }
  return static_cast<int>(poses.size()) - 1;
}


// Was bekommt dein Controller also konkret als path?
// Er bekommt:

// ✔ Eine geordnete Liste von Posen
// wie der Roboter laufen soll — typischerweise vom global planner erzeugt.

// ✔ Typischerweise im frame "map"
// weil globale Pfade meist in der Karte geplant werden.

// ✔ Jede Pose mit Zeitstempel + Pose-Daten

//Der aktuell verfolgte globale Pfad wird in current_path_ gespeichert.
//Wenn der Pfad keine Posen hat, wird abgebrochen (kein lokales Ziel möglich).

void RLController::setPlan(const nav_msgs::msg::Path & path) {
  current_path_ = path;
  if (current_path_.poses.empty()) return;

  const auto robot = robotPoseInMap_(); //Ermittelt zuerst die Robotpose (im Map-Frame).
  goal_idx_ = nearestIndexTo_(current_path_.poses, robot.pose.position);//Sucht dann den Index der Pose auf dem Pfad, die dem Roboter am nächsten ist (nearestIndexTo_).
  goal_idx_ = indexAtLookaheadFrom_(current_path_.poses, goal_idx_, lookahead_dist_);//der Punkt, der ungefähr lookahead_dist_ vor dem Roboter auf dem Pfad liegt.
  
//TODO:Hier ist der Fehler: die local costmap wird mit 1Hz aktualisiert und dadurch wird ständig ein neues local goal geschickt
  last_published_goal_ = current_path_.poses[goal_idx_];  //neues Goal bestimmen und publishen
  //hier umfüllen auf goalstate msg
  // turtlebot3_msgs::msg::GoalState goal_msg;
  // goal_msg.pose_x = last_published_goal_.pose.position.x;
  // goal_msg.pose_y = last_published_goal_.pose.position.y;

  global_path_pub->publish(current_path_);
  local_goal_pub_->publish(last_published_goal_);
//speichert den neuen globalen Pfad und setzt sofort ein erstes lokales Ziel ein Stück voraus auf dem Pfad (Lookahead), das veröffentlicht wird.
}

//bewegt das lokale Ziel dynamisch nach vorne,
//sobald der Roboter nah genug am aktuellen Ziel ist oder es überholt, und publiziert ein neues Ziel nur dann,
//wenn es sich ausreichend vom alten unterscheidet.


//TODO: Entweder hier wirklich NUR neues Goal wenn altes wirklich erreicht, unabhängig ob neuer globaler Pfad!!!
//ODER: Im BT einstellen, dass neuer globaler Pfad nicht mit 1Hz aktuallisiert wird!!!
void RLController::maybeAdvanceLocalGoal_() {
  if (current_path_.poses.empty()) return;

  const auto robot = robotPoseInMap_();

  // erreicht?
  //d_goal ist die Distanz vom letzten gepublishten Goal zur aktuellen Roboterpose
  const double d_goal = dist2_(last_published_goal_.pose.position, robot.pose.position);
  if(d_goal>reach_radius_) return;

  //wenn kleiner als definierter Radius
  
  const int near_idx = nearestIndexTo_(current_path_.poses, robot.pose.position);// findet den index der nähesten Pose im Pfad zum Roboter
  const int new_idx = indexAtLookaheadFrom_(current_path_.poses, near_idx, lookahead_dist_); //findet den Index der Pose im Pfad, welche lookadhead dist weit entfernt
  //von der Pose ist, die am nähsten zum Roboter ist
  // if (new_idx > goal_idx_) goal_idx_ = next_idx;//sicherstellen dass nicht rückwärts weglassen??
  //   {
  //   new_idx = goal_idx_ + 1;
  //   if (new_idx >= static_cast<int>(current_path_.poses.size())) {
  //     new_idx = static_cast<int>(current_path_.poses.size()) - 1;
  //   }
  //   }
  const auto & candidate = current_path_.poses[new_idx];
  if (dist2_(candidate.pose.position, last_published_goal_.pose.position) > advance_hysteresis_) return;

  goal_idx_ = new_idx;
  last_published_goal_ = candidate;

  // turtlebot3_msgs::msg::GoalState goal_msg;
  // goal_msg.pose_x = last_published_goal_.pose.position.x;
  // goal_msg.pose_y = last_published_goal_.pose.position.y;
  local_goal_pub_->publish(last_published_goal_);
  
}

geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(const geometry_msgs::msg::PoseStamped & /*pose*/,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  // Lokales Ziel ggf. vorziehen / neu publizieren
  maybeAdvanceLocalGoal_();

  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = node_->now();
  cmd.header.frame_id = costmap_ros_->getBaseFrameID();

  if (!client_ || !client_->service_is_ready()) {
    cmd.twist.linear.x = 0.0; cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto req = std::make_shared<turtlebot3_msgs::srv::RLLocalPath::Request>();
  req->stamp = node_->now();
  req->max_vx = static_cast<float>(max_lin_);
  req->max_wz = static_cast<float>(max_ang_);

  auto future = client_->async_send_request(req);
  if (future.wait_for(std::chrono::milliseconds(timeout_ms_)) != std::future_status::ready) {
    cmd.twist.linear.x = 0.0; cmd.twist.angular.z = 0.0;
    return cmd;
  }
  auto res = future.get();
  if (!res->ok) {
    cmd.twist.linear.x = 0.0; cmd.twist.angular.z = 0.0;
    return cmd;
  }

  auto clamp = [](double v, double lo, double hi){ return std::max(lo, std::min(hi, v)); };
  cmd.twist.linear.x  = clamp(res->vx, -max_lin_, max_lin_);
  cmd.twist.linear.y  = 0.0;
  cmd.twist.angular.z = clamp(res->wz, -max_ang_, max_ang_);
  return cmd;
}

void RLController::setSpeedLimit(const double & speed_limit, const bool & percentage) {
  if (speed_limit <= 0.0) { max_lin_ = 0.0; return; }
  if (percentage) {
    const double base = 0.22;
    max_lin_ = base * speed_limit / 100.0;
  } else {
    max_lin_ = speed_limit;
  }
}

} // namespace nav2_rl

PLUGINLIB_EXPORT_CLASS(nav2_rl::RLController, nav2_core::Controller)
