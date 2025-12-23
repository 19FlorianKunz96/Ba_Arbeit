#include "nav2_rl/rl_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <algorithm>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // doTransform
using namespace std::chrono_literals;

namespace nav2_rl {
// Soll nur noch globalen pfad schicken und kommandos als service anfordern, rest im agenten node
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
  
  auto declare_if_not_declared =
  [this](const std::string & full_name, const auto & default_value) {
    using T = std::decay_t<decltype(default_value)>;
    if (!node_->has_parameter(full_name)) {
      node_->declare_parameter<T>(full_name, default_value);
    }
    return node_->get_parameter(full_name).get_value<T>();
  };

agent_service_      = declare_if_not_declared(name_ + ".agent_service", std::string("/agents/dqn/get_action"));
timeout_ms_         = declare_if_not_declared(name_ + ".timeout_ms", 150);
max_lin_            = declare_if_not_declared(name_ + ".max_lin_vel", 0.22);
max_ang_            = declare_if_not_declared(name_ + ".max_ang_vel", 1.5);
lookahead_dist_     = declare_if_not_declared(name_ + ".lookahead_dist", 1.0);
reach_radius_       = declare_if_not_declared(name_ + ".reach_radius", 0.20);
advance_hysteresis_ = declare_if_not_declared(name_ + ".advance_hysteresis", 0.05);

  
  /*

  agent_service_     = node_->declare_parameter<std::string>(name_ + ".agent_service", "/agents/dqn/get_action");
  timeout_ms_        = node_->declare_parameter<int>(name_ + ".timeout_ms", 150); //so lange wird auf eine Antwort des Servers gewartet
  max_lin_           = node_->declare_parameter<double>(name_ + ".max_lin_vel", 0.22);
  max_ang_           = node_->declare_parameter<double>(name_ + ".max_ang_vel", 1.5);
  lookahead_dist_    = node_->declare_parameter<double>(name_ + ".lookahead_dist", 1.0); //wie weit soll der lokale zielpunkt entlang des globalen pfades weg sein
  reach_radius_      = node_->declare_parameter<double>(name_ + ".reach_radius", 0.20); //Radius um Ziel, in dem es als erreicht gilt und neues lokales Goal gepublisht wird
  advance_hysteresis_= node_->declare_parameter<double>(name_ + ".advance_hysteresis", 0.05); //Mindestverschiebung, um ein neues lokales Ziel zu publizieren 
  //(damit nicht bei kleinsten Änderungen neu gesendet wird). Ist das sinnvoll ? Dann wird ja wenn das Ziel weniger als das weg liegt, nichts mehr gepublisht??? */

  // Publisher für lokales Ziel
  rclcpp::QoS qos(1); qos.transient_local().reliable();
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

//Globalen Pfad erhalten(beim Setzen des Ziels oder beim Replanning) und an Agenten Node schicken. Weitere Berechnung der Teilpfade dort.
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
void RLController::setPlan(const nav_msgs::msg::Path & path) {
  current_path_ = path;
  if (current_path_.poses.empty()) return;


  nav_msgs::msg::Path path_odom;
  path_odom.header = path.header;
  path_odom.header.frame_id = "odom";

  path_odom.poses.reserve(path.poses.size());
  for (auto pose : path.poses) {
    try {
      auto tf_msg = tf_->lookupTransform("odom", pose.header.frame_id, tf2::TimePointZero);
      tf2::doTransform(pose, pose, tf_msg);
      pose.header.frame_id = "odom";
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(node_->get_logger(), "TF map->odom failed: %s", e.what());
    }
    path_odom.poses.push_back(pose);
  }

  global_path_pub->publish(path_odom);

}

// Service anfordern um Kommandos zu bekommen
geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(const geometry_msgs::msg::PoseStamped & /*pose*/,
  const geometry_msgs::msg::Twist & ,
  nav2_core::GoalChecker * )
{

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
  cmd.twist.linear.x  = res->vx;
  cmd.twist.linear.y  = 0.0;
  cmd.twist.angular.z = res->wz;
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
