#!/usr/bin/env python
import rospy
import roboy_dep.msg

import numpy as np
import time

def initDepParams():
	depParams = roboy_dep.msg.depParameters()
	depParams.timedist = 8
	depParams.urate = 0.01
	depParams.initFeedbackStrength = 0
	depParams.regularization = 10
	depParams.synboost = 1.7
	depParams.maxSpeed = 1
	depParams.epsh = 0.0
	depParams.pretension = 5 
	depParams.maxForce = 300
	depParams.springMult1 = 1.0
	depParams.springMult2 = 1.0
	depParams.diff = 1
	depParams.amplitude = 1.0
	depParams.delay = 50
	depParams.guideType = 0 
	depParams.guideIdx = 0
	depParams.guideAmpl = 0.3
	depParams.guideFreq = 0.5
	depParams.learning = False
	depParams.targetForce = 20
	depParams.range = 5
	return depParams

def run_script(msg):
	time.sleep(0.1)

	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = True
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 21.
	brain_id_pub.publish(brain_id)
	time.sleep(1.0)

	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = False
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 42.
	brain_id_pub.publish(brain_id)
	time.sleep(1.0)

	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = False
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 21.
	brain_id_pub.publish(brain_id)
	time.sleep(1.0)

	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = True
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 42.
	brain_id_pub.publish(brain_id)
	time.sleep(1.0)

	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = True
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 0.
	brain_id_pub.publish(brain_id)
	time.sleep(1.0)

def main():
	rospy.init_node('brain', anonymous=True)

	global brain_id_pub
	global tran_dir_pub
	global stop_pub
	brain_id_pub = rospy.Publisher('/roboy_dep/brain_id', roboy_dep.msg.brain_id,queue_size=1)
	tran_dir_pub = rospy.Publisher('/roboy_dep/transition_direction', roboy_dep.msg.transition_direction,queue_size=1)
	stop_pub = rospy.Publisher('/roboy_dep/stop', roboy_dep.msg.stop,queue_size=1)

	global depParams
	global dep_param_pub
	depParams = initDepParams()
	dep_param_pub = rospy.Publisher('/roboy_dep/depParameters', roboy_dep.msg.depParameters,queue_size=1)

	rospy.Subscriber("/roboy_dep/start_script", roboy_dep.msg.start_script, run_script, queue_size = 1)

	rospy.spin()

'''
	velocity = 60
	amplitude = 0.8
	depParams.delay = velocity
	depParams.amplitude = amplitude
	dep_param_pub.publish(depParams)

	stop = roboy_dep.msg.stop
	stop.stop = True
	stop_pub.publish(stop)
'''

if __name__ == '__main__':
	main()