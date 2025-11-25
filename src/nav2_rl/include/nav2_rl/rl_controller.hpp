#pragma once
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"
#include "turtlebot3_msgs/srv/rl_local_path.hpp"
#include "turtlebot3_msgs/msg/goal_state.hpp"

namespace nav2_rl {

class RLController : public nav2_core::Controller {
public:
  RLController() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override {}
  void activate() override {}
  void deactivate() override {}

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  // Node & Nav2
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string name_;

  // Params
  std::string agent_service_;
  int timeout_ms_{150};
  double max_lin_{0.22};
  double max_ang_{1.5};

  // Service client
  rclcpp::Client<turtlebot3_msgs::srv::RLLocalPath>::SharedPtr client_;

  // Optional: Pfad / Lokales Ziel
  nav_msgs::msg::Path current_path_;
  int goal_idx_{0};
  double lookahead_dist_{1.0};
  double reach_radius_{0.2};
  double advance_hysteresis_{0.05};

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr local_goal_pub_;
  geometry_msgs::msg::PoseStamped last_published_goal_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub;

  // Param-Callback
  rclcpp_lifecycle::LifecycleNode::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // Helper (Deklaration)
  void connectClient_();
   geometry_msgs::msg::PoseStamped robotPoseInMap_();

};

} // namespace nav2_rl
