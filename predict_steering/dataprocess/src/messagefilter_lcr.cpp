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


void callback ( const sensor_msgs::ImageConstPtr& limage,
			    const sensor_msgs::ImageConstPtr& cimage,
			    const sensor_msgs::ImageConstPtr& rimage, 
			    const prius_msgs::SteeringAngleConstPtr& steering,
				const ros::NodeHandle nh,
				ros::Publisher left_pub,
				ros::Publisher center_pub,
				ros::Publisher right_pub,
				ros::Publisher steering_pub	)
{	

	left_pub.publish(limage);
	center_pub.publish(cimage);
	right_pub.publish(rimage);	
	steering_pub.publish(steering);
}



int main ( int argc, char** argv)
{
	ros::init(argc, argv, "filter_node");
	ros::NodeHandle nh;
	
	message_filters::Subscriber<sensor_msgs::Image> cimage_sub(nh, "/prius/front_camera/image_raw",1);
	message_filters::Subscriber<sensor_msgs::Image> limage_sub(nh, "/prius/left_camera/image_raw", 1);
	message_filters::Subscriber<sensor_msgs::Image> rimage_sub(nh, "prius/right_camera/image_raw" , 1);
	message_filters::Subscriber<prius_msgs::SteeringAngle> steering_sub(nh, "/prius/steering_angle" , 1);

	ros::Publisher cimage_pub = nh.advertise<sensor_msgs::Image>("/filtered/center/image_raw", 1);	
	ros::Publisher limage_pub = nh.advertise<sensor_msgs::Image>("/filtered/left/image_raw", 1);	
	ros::Publisher rimage_pub = nh.advertise<sensor_msgs::Image>("/filtered/right/image_raw", 1);
	ros::Publisher steering_pub = nh.advertise<prius_msgs::SteeringAngle>("/filtered/steering_angle" ,1);

	TimeSynchronizer<sensor_msgs::Image , sensor_msgs::Image , sensor_msgs:: Image , prius_msgs::SteeringAngle> sync(limage_sub, cimage_sub, rimage_sub, steering_sub, 10);

	sync.registerCallback(boost::bind(&callback , _1, _2 ,_3, _4, nh, limage_pub, cimage_pub, rimage_pub, steering_pub));
	

  	ros::spin();

  	return 0;

}