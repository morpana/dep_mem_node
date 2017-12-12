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

def behavior_loaded(msg):
	global next_behavior
	next_behavior = True

def vel_script(msg):
	time.sleep(0.1)
	velocity = 30
	amplitude = 1.0
	depParams.amplitude = amplitude
	depParams.delay = velocity
	dep_param_pub.publish(depParams)
	while(velocity<90):
		velocity += 1
		depParams.delay = velocity
		dep_param_pub.publish(depParams)
		time.sleep(0.04)

def run_script(msg):
	global next_behavior
	time.sleep(0.1)

	velocity = 80
	amplitude = 0.70
	depParams.amplitude = amplitude
	depParams.delay = velocity
	#duration = 0.2
	dep_param_pub.publish(depParams)
	for i in range(5):
		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = False
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 42.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior):
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		'''
		stop = roboy_dep.msg.stop()
		stop.stop = True
		stop_pub.publish(stop)
		time.sleep(duration)
		stop.stop = False
		stop_pub.publish(stop)
		'''

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = False
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 21.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior): 
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = True
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 42.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior): 
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = True
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 21.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior):
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		velocity -= 4
		#duration -= 0.02
		amplitude += 0.05
		depParams.delay = velocity
		dep_param_pub.publish(depParams)

		'''
		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = True
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 0.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior): 
			print "sleeping"
			time.sleep(0.1)
		'''
		print "\n"

	for i in range(3):
		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = False
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 42.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior):
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = False
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 21.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior): 
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = True
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 42.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior): 
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)

		tran_dir = roboy_dep.msg.transition_direction()
		tran_dir.up = True
		tran_dir_pub.publish(tran_dir)
		brain_id = roboy_dep.msg.brain_id()
		brain_id.brain_id = 21.
		brain_id_pub.publish(brain_id)
		print "brain_id: ", brain_id
		next_behavior = False
		while(not next_behavior):
			print "sleeping"
			time.sleep(0.1)
		time.sleep(0.5)


	tran_dir = roboy_dep.msg.transition_direction()
	tran_dir.up = False
	tran_dir_pub.publish(tran_dir)
	brain_id = roboy_dep.msg.brain_id()
	brain_id.brain_id = 0.
	brain_id_pub.publish(brain_id)
	print "brain_id: ", brain_id

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

	global next_behavior
	next_behavior = True

	rospy.Subscriber('/roboy_dep/depLoadMatrix', roboy_dep.msg.depMatrix, behavior_loaded, queue_size = 1)

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