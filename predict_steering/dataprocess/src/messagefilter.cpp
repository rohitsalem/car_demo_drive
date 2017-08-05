#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <prius_msgs/SteeringAngle.h>
#include <ros/ros.h>

using namespace sensor_msgs;
using namespace message_filters;
using namespace std;


void callback (const sensor_msgs::ImageConstPtr& image, const prius_msgs::SteeringAngleConstPtr& steering, const ros::NodeHandle nh,ros::Publisher steering_pub,ros::Publisher image_pub)
{	

	sensor_msgs::Image image_msg;
	prius_msgs::SteeringAngle steering_msg;

	steering_msg.data = steering->data;
	steering_msg.header = steering->header;

	steering_pub.publish(steering_msg);
	image_pub.publish(image);
	
}



int main ( int argc, char** argv)
{
	ros::init(argc, argv, "filter_node");
	ros::NodeHandle nh;
	message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/prius/front_camera/image_raw",1);
	message_filters::Subscriber<prius_msgs::SteeringAngle> steering_sub(nh,"/prius/steering_angle",1);
	ros::Publisher steering_pub = nh.advertise<prius_msgs::SteeringAngle>("/filtered/steering_angle", 1);
	ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/filtered/image_raw", 1);	

	// typedef sync_policies::ApproximateTime <sensor_msgs::Image, prius_msgs::SteeringAngle> MysyncPolicy;

	// Synchronizer<MysyncPolicy> sync (MysyncPolicy(10), image_sub, steering_sub);

	TimeSynchronizer<sensor_msgs::Image , prius_msgs::SteeringAngle> sync(image_sub,steering_sub,10);

	sync.registerCallback(boost::bind(&callback , _1, _2 ,nh, steering_pub, image_pub));
	

  	ros::spin();

  	return 0;

}