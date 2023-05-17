''' Library with different functions for assesing predicion performance on several levels:
1. event level (for. example, seizure episode level), not caring about the exact length overalp between tru and predicted event,
it classifies a match if there is any overlap between predicted and true event
2. duration level (or sample by sample level) is classical performance metric that cares about each sample classification
3. combinations of event and duration metrics (e.g. mean or geo mean of F1 scores of event and duration based metrics)
4. number of false positives per day - useful for biomedical applications (such as epilepsy monitoring) '''

__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class EventsAndDurationPerformances ():
	''' Set of functions to measure performance of ML models not on a sample-by-sample basis, 
	but on a basis of events/episodes (sequences of the same labels). It currently works only for binary classification. 
	'''
	def __init__(self, PerfParams): 
		self.samplFreq=PerfParams.samplFreq
		self.numLabelsPerHour= 60 * 60 *self.samplFreq #number of samples per hour, needed to estimate numFP/hour
		self.toleranceFP_bef=int(PerfParams.toleranceFP_befEvent *self.samplFreq)  #how many samples before the event is still ok to be event without saying that it is false positive
		self.toleranceFP_aft=int(PerfParams.toleranceFP_aftEvent * self.samplFreq) #how many samples after the event is still ok to be event without saying that it is false positive

		self.movingWinLen=PerfParams.movingWinLen #window in which it smooths, in seconds
		self.movingWinLenIndx=int(PerfParams.movingWinLen * self.samplFreq) #the same as movingWinLen but in num of samples
		self.movingWinPercentage= PerfParams.movingWinPercentage #what percentage of labels needs to be 1 to say that it is 1
		self.distanceBetween2events=PerfParams.distanceBetween2events  #if events are closer then distanceBetweenEventsIndx then it merges them to one (puts all labels inbetween to 1 too), in seconds
		self.distanceBetweenEventsIndx =int(PerfParams.distanceBetween2events* self.samplFreq) #the same as distanceBetween2events but in num of samples

		self.bayesProbThresh=PerfParams.bayesProbThresh #bayes threshold

	
	def calculateStartsAndStops(self, labels):
		''' Function that detects starts and stop of event (or groups of labels 1) '''
		sigLen = len(labels)
		events = []
		for i in range(1, sigLen - 1):
			# if  label event starts
			if (labels[i] == 1 and labels[i - 1] == 0) | (i == 1 and labels[i-1] == 1 and labels[i] == 1):
				sstart = i
				while labels[i] == 1 and i < sigLen - 1:
					i = i + 1
				sstop = i
				events.append([sstart, sstop])
		return (events)
	
	def calc_TPAndFP(self, ref, hyp):
		''' For a pair of ref and hyp event decides if it is false or true prediction '''

		## collect start and stop times from input arg events
		start_ref = ref[0]
		stop_ref = ref[1]
		start_hyp = hyp[0]
		stop_hyp = hyp[1]

		##### detect if hyp and ref have some overlap
		tp = 0

		#     ref:            |        <--------------------->
		#     hyp:     ---------->
		if (start_hyp <= start_ref and stop_hyp > start_ref - self.toleranceFP_bef):  # + tolerance
			tp = 1
		#     ref:              <--------------------->       |
		#     hyp:                                        <----------------
		elif (start_hyp < stop_ref + self.toleranceFP_aft and stop_hyp >= stop_ref):  # - tolerance
			tp = 1
		#     ref:              <--------------------->
		#     hyp:                         <---->
		elif (stop_hyp <= stop_ref and start_hyp >= start_ref):  # - tolerance
			tp = 1

		#### detect fp
		fp = 0
		fp_bef = 0
		fp_aft = 0
		#     ref:         |     <--------------------->     |
		#     hyp:     <----------------
		if (start_hyp < start_ref - self.toleranceFP_bef):
			fp = fp + 1
			fp_bef = 1
		#     ref:         |     <--------------------->     |
		#     hyp:    						  ------------------>
		if (stop_hyp > stop_ref + self.toleranceFP_aft):
			fp = fp + 1
			fp_aft = 1
		return (tp, fp, fp_bef, fp_aft)
		
	#
	# def performance_events_old(self, predLab, trueLab):
	# 	'''
	# 	Function that detects events in a stream of true labels and predictions
	# 	Detects overlaps and measures sensitivity, precision , F1 score and number of false positives
	# 	'''
	# 	totalTP = 0
	# 	totalFP = 0
	# 	# transform to events
	# 	predEvents = self.calculateStartsAndStops(predLab)
	# 	trueEvents = self.calculateStartsAndStops(trueLab)
	#
	# 	# create flags for each event if it has been used
	# 	flag_predEvents = np.zeros(len(predEvents))
	# 	flag_trueEvents = np.zeros(len(trueEvents))
	# 	flag_trueEventsFPAround = np.zeros(len(trueEvents))
	# 	# goes through ref events
	# 	if (len(trueEvents) == 0):
	# 		totalFP = len(predEvents)
	# 	else:
	# 		for etIndx, eTrue in enumerate(trueEvents):
	# 			for epIndx, ePred in enumerate(predEvents):
	# 				(tp0, fp0, fp_bef, fp_aft) = self.calc_TPAndFP(eTrue, ePred)
	#
	# 				# if overlap detected (tp=1) and this refEvent hasnt been used
	# 				#     ref:           <----->        <----->              <----->             <-------------->
	# 				#     hyp:     <---------->          <---------->     <-------------->           <----->
	# 				if (tp0 == 1 and flag_trueEvents[etIndx] == 0 and flag_predEvents[epIndx] == 0):
	# 					totalTP = totalTP + tp0
	# 					totalFP = totalFP + fp0
	# 					flag_trueEvents[etIndx] = 1
	# 					flag_predEvents[epIndx] = 1  # 1 means match
	# 					if (fp0 == 2):
	# 						flag_trueEventsFPAround[etIndx] = 2
	# 					else:
	# 						flag_trueEventsFPAround[etIndx] = fp_aft - fp_bef  # 1 if was after, or -1 if was before
	# 				# if ref event was already matched and now we have some extra predicted event ones
	# 				#     ref:           <------------------------------------------>
	# 				#     hyp:     <---------->     <-------------->     <----->
	# 				elif (tp0 == 1 and flag_trueEvents[etIndx] == 1 and flag_predEvents[epIndx] == 0):
	# 					# totalTP = totalTP + tp0 #we already counted that one
	# 					totalFP = totalFP + fp0  # ideally fp0 should be 0, but if at the end we might have 1 fp
	# 					# flag_trueEvents[etIndx] = 1 it is already 1 so not needed again
	# 					flag_predEvents[epIndx] = 2  # 2 means overlapping but not the first match with seizure
	# 					# update flag_trueEventsFPAround if needed
	# 					if (flag_trueEventsFPAround[etIndx]==-1 and fp_aft==1):
	# 						flag_trueEventsFPAround[etIndx]=2 #if suddenly fp is now on both sides of seizure
	# 					elif (flag_trueEventsFPAround[etIndx]==0 and fp_aft==1):
	# 						flag_trueEventsFPAround[etIndx] = 1  # if before was within borders but now is too long after seizure
	#
	# 				# if one big pred event covering more ref
	# 				#     ref:         <---------->     <-------------->     <----->
	# 				#     hyp:              <------------------------------------------>
	# 				# elif (tp0 == 1 and flag_trueEvents[etIndx] == 0 and flag_predEvents[epIndx] == 1):
	# 				# 	# totalTP=totalTP+tp0 # HERE WE NEED TO DECIDE TO WE COUNT THEM AS TP OR NOT  !!!!
	# 				# 	totalFP = totalFP + fp0  # we treat this as 1 FP
	# 				# 	if (flag_trueEventsFPAround[etIndx - 1] > 0 and fp_bef == 1):  # if there was FP after true event and already counted and now we want to count it again because of before
	# 				# 		totalFP = totalFP - 1
	# 				# 	flag_trueEvents[etIndx] = 0  # it has to stay unmatched
	# 				# 	# flag_predEvents[epIndx] = 1 #already matched
	# 				elif (tp0 == 1 and flag_trueEvents[etIndx] == 0 and flag_predEvents[epIndx] == 1):
	# 					totalTP=totalTP+tp0 # HERE WE NEED TO DECIDE TO WE COUNT THEM AS TP OR NOT  !!!!
	# 					flag_trueEvents[etIndx] = 1  # HERE WE NEED TO DECIDE TO WE COUNT THEM AS TP OR NOT  !!!!
	# 					# flag_predEvents[epIndx] = 1 #already matched
	# 					#update number of FP
	# 					#     ref:         <---------->     <-------------->
	# 					#     hyp:              <---------------------->
	# 					if (flag_trueEventsFPAround[etIndx - 1]==1 and fp_bef == 1):
	# 						#dont change number of totalFP because thing in between them already counted as FP
	# 						flag_trueEventsFPAround[etIndx] = -1
	# 					#     ref:         <---------->     <-------------->
	# 					#     hyp:       <----------------------------->
	# 					if (flag_trueEventsFPAround[etIndx - 1] == 2 and fp_bef == 1):
	# 						# dont change number of totalFP because thing in between them already counted as FP
	# 						flag_trueEventsFPAround[etIndx] = -1
	# 					#     ref:         <---------->     <-------------->
	# 					#     hyp:               <----------------------------->
	# 					if (flag_trueEventsFPAround[etIndx - 1] == 1 and fp_aft == 1):
	# 						totalFP = totalFP + 1
	# 						flag_trueEventsFPAround[etIndx] = 2
	# 					#     ref:         <---------->     <-------------->
	# 					#     hyp:       <-------------------------------------->
	# 					if (flag_trueEventsFPAround[etIndx - 1] == 2 and fp_aft == 1):
	# 						totalFP = totalFP + 1
	# 						flag_trueEventsFPAround[etIndx] = 2
	#
	#
	#
	#
	# 				# if pred event was named FP in pass with previous event but now matches this event
	# 				# elif (tp0 == 1 and flag_trueEvents[etIndx] == 0 and flag_predEvents[epIndx] == -1):
	# 				# 	totalTP = totalTP + tp0
	# 				# 	totalFP = totalFP - 1 + fp0  # remove fp from before
	# 				# 	flag_trueEvents[etIndx] = 1  # match event
	# 				# 	if (fp0 == 2):
	# 				# 		flag_trueEventsFPAround[etIndx] = 2
	# 				# 	else:
	# 				# 		flag_trueEventsFPAround[etIndx] = fp_aft - fp_bef  # 1 if was after, or -1 if was before
	# 				# 	flag_predEvents[epIndx] = 1  # relabel this pred event
	# 				elif (tp0 == 1 and flag_trueEvents[etIndx] == 0 and flag_trueEventsFPAround[epIndx-1] >=1 ):
	# 					totalTP = totalTP + tp0
	# 					totalFP = totalFP - 1 + fp_aft  # remove fp from before and add if lats too long
	# 					flag_trueEvents[etIndx] = 1  # match event
	# 					flag_predEvents[epIndx] = 1  # relabel this pred event
	# 					if (fp0 == 2):
	# 						flag_trueEventsFPAround[etIndx] = 2
	# 					else:
	# 						flag_trueEventsFPAround[etIndx] = fp_aft - fp_bef  # 1 if was after, or -1 if was before
	#
	#
	#
	# 				# if pred event was named FP in pass with previous event , now overlaps with event but this event was already matched
	# 				elif (tp0 == 1 and flag_trueEvents[etIndx] == 1 and flag_predEvents[epIndx] == -1):
	# 					# totalTP = totalTP + tp0 #we already counted that one
	# 					totalFP = totalFP - 1 + fp0  # ideally fp0 should be 0, but if at the end we might have 1 fp, remove Fp from before
	# 					# flag_trueEvents[etIndx] = 1 it is already 1 so not needed again
	# 					flag_predEvents[epIndx] = 2  # 2 means overlapping but not the first match with seizure
	# 				# prdiction but not matched with true event
	# 				elif (tp0 == 0 and flag_predEvents[epIndx] == 0):
	# 					totalFP = totalFP + 1  # +fp0
	# 					flag_predEvents[epIndx] = -1  # -1 means used as FP
	# 				elif (flag_predEvents[epIndx] == 2):  # already counted as part of previous event, we dont need to count it again
	# 					a = 0
	# 				elif (flag_predEvents[epIndx] == -1):  # already counted as FP, we dont need to count it again
	# 					a = 0
	# 				elif (flag_trueEvents[etIndx] == 1 and flag_predEvents[epIndx] == 1):  # both already matched
	# 					a = 0
	# 				elif (flag_trueEvents[etIndx] == 0 and flag_predEvents[epIndx] == 1):  # pred event was matched, true hasnt found yet
	# 					a = 0
	# 				else:
	# 					# flag_predEvents[epIndx] = 1
	# 					print('ERROR: new case I havent covered')
	#
	# 	# calculating final performance
	# 	numTrueEvent = len(trueEvents)
	# 	numPredEvent = len(predEvents)
	#
	# 	numMissedEvent = numTrueEvent - np.sum(flag_trueEvents)
	#
	# 	# precision =TP/ numPredEvent but if all is one big predicted event then thigs are wrong and value would be >1
	# 	if ((totalTP + totalFP) != 0):
	# 		precision = totalTP / (totalTP + totalFP)
	# 	else: #if only 0 predicted the whole time
	# 		precision = np.nan
	# 		F1score = np.nan
	# 		#sensitivity will be 0 in next if (if there are trueEvents)
	#
	# 	# sensitivity= TP/ numTrueSeiz
	# 	if ((numTrueEvent) != 0):
	# 		sensitivity = totalTP / numTrueEvent
	# 	else: #in case no true seizures in a file
	# 		sensitivity = np.nan
	# 		precision = np.nan
	# 		F1score = np.nan
	#
	#
	# 	if ((sensitivity + precision) != 0):
	# 		F1score = (2 * sensitivity * precision) / (sensitivity + precision)
	# 	else: #if somehow there was no TP for senstivity and precision are 0
	# 		F1score = np.nan  # 0 - maybe it should be 0 ??
	#
	# 	# checkups
	# 	# if ( (totalTP +totalFP)!= numPredEvent):
	# 	#     print('there are long pred events')
	# 	if ((numMissedEvent + totalTP) != numTrueEvent):
	# 		print('sth wrong with counting events')
	# 	if (totalFP < len(np.where(flag_predEvents == -1)[0])):
	# 		print('sth wrong with counting FP')
	# 	if (totalTP != len(np.where(flag_predEvents == 1)[0])):
	# 		print('sth wrong with counting true events')
	#
	# 	# #visualize
	# 	# xvalues = np.arange(0, len(trueLab), 1)
	# 	# fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
	# 	# gs = GridSpec(1, 1, figure=fig1)
	# 	# fig1.subplots_adjust(wspace=0.4, hspace=0.6)
	# 	# fig1.suptitle('True and pred labels')
	# 	# ax1 = fig1.add_subplot(gs[0,0])
	# 	# ax1.plot(xvalues, trueLab,  'r')
	# 	# ax1.plot(xvalues, predLab*0.9, 'k')
	# 	# ax1.set_xlabel('Time')
	# 	# ax1.legend(['True', 'Pred'])
	# 	# #calcualte performance for duration to put in title
	# 	# (sensitivity_duration, precision_duration, F1score_duration) = perfomance_duration(predLab, trueLab)
	# 	# ax1.set_title('EVENTS: sensitivity '+ str(sensitivity)+' precision ' + str(precision)+' F1score '+ str(F1score)+' totalTP ' + str(totalTP)+ ' totalFP '+ str(totalFP) + '\n' +
	# 	#               'DURATION: sensitivity ' + str(sensitivity_duration) + ' precision ' + str(precision_duration) + ' F1score ' + str(F1score_duration))
	# 	# ax1.grid()
	# 	# fig1.show()
	# 	# fig1.savefig(folderOut + '/' +figName)
	# 	# plt.close(fig1)
	#
	# 	return (sensitivity, precision, F1score, totalFP)

	def performance_events(self, predLab, trueLab):
		'''
		Function that detects events in a stream of true labels and predictions
		Detects overlaps and measures sensitivity, precision , F1 score and number of false positives
		'''
		#################################
		## TRANSFORMING  BINARY LABELS TO EVENTS
		predEvents = self.calculateStartsAndStops(predLab)
		trueEvents = self.calculateStartsAndStops(trueLab)

		#################################
		## MATCHING EVENTS
		# create flags for each event if it has been used
		flag_predEvents = np.zeros(len(predEvents))
		flag_trueEvents = np.zeros(len(trueEvents))
		flag_predEventsFPAround = np.zeros(len(predEvents))
		# goes through ref events
		if (len(trueEvents) == 0):
			totalFP = len(predEvents)
		else:
			for etIndx, eTrue in enumerate(trueEvents):
				for epIndx, ePred in enumerate(predEvents):
					(tp0, fp0, fp_bef, fp_aft) = self.calc_TPAndFP(eTrue, ePred)

					#if overlap detected
					if (tp0==1):
						flag_trueEvents[etIndx] = 1
						flag_predEvents[epIndx] = flag_predEvents[epIndx]+ 1

						if (flag_predEvents[epIndx]>1): #already had FP before around matched pred event
							if (flag_predEventsFPAround[epIndx]==-1 and fp_aft==1):
								flag_predEventsFPAround[epIndx]==2 #mark that it has FP on both sides
						else:
							if (fp0 == 2):
								flag_predEventsFPAround[epIndx] = 2
							else:
								flag_predEventsFPAround[epIndx] = fp_aft - fp_bef  # 1 if was after, or -1 if was before

					# if no overlap, we will count it later as FP

		#################################
		## COUNTING EVENTS - TOTAL NUMBER OF TP, FP, FN etc
		numTrueEvent = len(trueEvents)
		numPredEvent = len(predEvents)

		totalTP= sum(flag_trueEvents) #count how many true events was labeled as matched
		# count FP
		numNonMatchedPredictedEvents= len(np.where(flag_predEvents==0)[0])
		numFParoundMatched=np.sum(np.abs(flag_predEventsFPAround))

		helper=flag_predEvents - np.ones(len(flag_predEvents))
		numFPbetweenLongPredEvents=np.sum( helper[np.where(helper>0)])
		totalFP=numNonMatchedPredictedEvents+numFParoundMatched+numFPbetweenLongPredEvents

		numMissedEvent = numTrueEvent - np.sum(flag_trueEvents)

		#################################
		## CALCUALTING PERFORMANCE METRICS
		# precision =TP/ numPredEvent but if all is one big predicted event then things are wrong and value would be >1
		if ((totalTP + totalFP) != 0):
			precision = totalTP / (totalTP + totalFP)
		else:  # if only 0 predicted the whole time
			precision = np.nan
			F1score = np.nan
		# sensitivity will be 0 in next if (if there are trueEvents)

		# sensitivity= TP/ numTrueSeiz
		if ((numTrueEvent) != 0):
			sensitivity = totalTP / numTrueEvent
		else:  # in case no true seizures in a file
			sensitivity = np.nan
			precision = np.nan
			F1score = np.nan

		if ((sensitivity + precision) != 0):
			F1score = (2 * sensitivity * precision) / (sensitivity + precision)
		else:  # if somehow there was no TP for senstivity and precision are 0
			F1score = np.nan  # 0 - maybe it should be 0 ??

		# checkups
		# if ( (totalTP +totalFP)!= numPredEvent):
		#     print('there are long pred events')
		if ((numMissedEvent + totalTP) != numTrueEvent):
			print('sth wrong with counting events')
		if (totalFP < len(np.where(flag_predEvents == -1)[0])):
			print('sth wrong with counting FP')
		if (totalTP != len(np.where(flag_predEvents == 1)[0])):
			print('sth wrong with counting true events')

		# #visualize
		# xvalues = np.arange(0, len(trueLab), 1)
		# fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
		# gs = GridSpec(1, 1, figure=fig1)
		# fig1.subplots_adjust(wspace=0.4, hspace=0.6)
		# fig1.suptitle('True and pred labels')
		# ax1 = fig1.add_subplot(gs[0,0])
		# ax1.plot(xvalues, trueLab,  'r')
		# ax1.plot(xvalues, predLab*0.9, 'k')
		# ax1.set_xlabel('Time')
		# ax1.legend(['True', 'Pred'])
		# #calcualte performance for duration to put in title
		# (sensitivity_duration, precision_duration, F1score_duration) = perfomance_duration(predLab, trueLab)
		# ax1.set_title('EVENTS: sensitivity '+ str(sensitivity)+' precision ' + str(precision)+' F1score '+ str(F1score)+' totalTP ' + str(totalTP)+ ' totalFP '+ str(totalFP) + '\n' +
		#               'DURATION: sensitivity ' + str(sensitivity_duration) + ' precision ' + str(precision_duration) + ' F1score ' + str(F1score_duration))
		# ax1.grid()
		# fig1.show()
		# fig1.savefig(folderOut + '/' +figName)
		# plt.close(fig1)

		return (sensitivity, precision, F1score, totalFP)

	def performance_duration(self, y_pred_smoothed, y_true):
		'''Calculates performance metrics on the sample by sample basis '''

		# total true event durations
		durationTrueEvent = y_true.sum().item()

		# total predicted event duration
		durationPredictedEvent = y_pred_smoothed.sum().item()

		# total duration of true predicted event
		temp = 2 * y_true - y_pred_smoothed  # where diff is 1 here both true and apredicted label are 1
		durationTruePredictedEvent = len(temp[temp == 1])

		# Calculating sensitivity
		if (durationTrueEvent == 0): #no true event
			sensitivity = np.nan #0
			# print('No event in test data')
		else:
			sensitivity = durationTruePredictedEvent / durationTrueEvent

		# Calculating precision
		if (durationPredictedEvent == 0): #no predicted event
			precision = np.nan #0
			# print('No predicted event in test data')
		else:
			precision = durationTruePredictedEvent / durationPredictedEvent

		# Calculating F1score
		if ((sensitivity + precision) == 0): #in case no overlap in true and predicted events
			F1score_duration = 0
			# print('No overlap in predicted events')
		elif ((durationTrueEvent == 0) or (durationPredictedEvent == 0)):  #in case no true or predicted event
			F1score_duration= np.nan
			# print('No true or predicted events')
		else:
			F1score_duration = 2 * sensitivity * precision / (sensitivity + precision)

		return (sensitivity, precision, F1score_duration)


	def performance_all9(self, predLab, trueLab):
		''' Function that returns 9 different performance measures of prediction on epilepsy
		1. on the level of events (sensitivity, precision and F1 score)
		2. on the level of event duration, or each sample (sens, prec, F1 score)
		3. combination of F1 scores for events and duration ( mean or gmean)
		4. number of false positives per day
		Returns them in this order:  ['Sensitivity events', 'Precision events', 'F1score events', 'Sensitivity duration',
					 'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
		'''

		(sensE, precisE, F1E, totalFP) = self.performance_events(predLab, trueLab)
		(sensD, precisD, F1D) = self.performance_duration(predLab, trueLab)

		# calculate combinations
		F1DEmean = (F1D + F1E) / 2
		F1DEgeoMean = np.sqrt(F1D * F1E)

		# calculate numFP per day
		timeDurOfLabels = len(trueLab) / self.numLabelsPerHour
		if (timeDurOfLabels != 0):
			numFPperHour = totalFP / timeDurOfLabels
		else:
			numFPperHour = np.nan
		numFPperDay = numFPperHour * 24

		# last checkup
		if ( sensE > 1.0 or precisE > 1.0 or F1E > 1.0 or sensD > 1.0 or precisD > 1.0 or F1D > 1.0 or F1DEmean > 1.0 or F1DEgeoMean > 1.0):
			print('ERROR - perf measures impossibly big!')
		# if (np.sum(trueLab)==0):
		#     print('No Event in file')

		return torch.tensor((sensE, precisE, F1E, sensD, precisD, F1D, F1DEmean, F1DEgeoMean, numFPperDay))


	def smoothenLabels_movingAverage(self, prediction0):
		''' Returns labels after two steps of postprocessing
		First: moving window with voting (if more then threshold of labels are 1 final label is 1 otherwise 0)
		Second: merging events that are too close
		Parameters:
				- movingWinLenIndx - window in which it smooths
				- movingWinPercentage - what percentage of labels needs to be 1 to say that it is 1
				- distanceBetweenEventsIndx - finally, if events are closer then distanceBetweenEventsIndx then it merges them to one (puts all labels inbetween to 1 too)
		'''

		if not torch.is_tensor(prediction0): #if array passed
			prediction=torch.from_numpy(prediction0)
		else:
			prediction=prediction0

		p = torch.nn.ConstantPad1d((self.movingWinLenIndx - 1, 0), 0)
		# first classifying as true 1 if at laest  GeneralParams.movingWinLen in a row is 1
		unfolded_prediction = prediction.unfold(0, self.movingWinLenIndx, 1).float()
		smoothLabelsStep1 = p(torch.where(unfolded_prediction.mean(1) >= self.movingWinPercentage, 1, 0))

		# second part
		smoothLabelsStep2 = torch.clone(smoothLabelsStep1)
		# Calculate event starts and stops
		events = self.calculateStartsAndStops(smoothLabelsStep2)
		# if  event started but is too close to previous one, delete second event by connecting them
		for idx in range(1, len(events)):
			if events[idx][0] - events[idx - 1][1] <= self.distanceBetweenEventsIndx:
				smoothLabelsStep2[events[idx - 1][1]:events[idx][0]] = 1

		if not torch.is_tensor(prediction0):
			return (smoothLabelsStep1.numpy(), smoothLabelsStep2.numpy())
		else:
			return (smoothLabelsStep1, smoothLabelsStep2)


	def smoothenLabels_Bayes(self, prediction0, probability0):
		''' Returns labels bayes postprocessing
		Calculates cummulative probability of event and non event over the window of size movingWinLenIndx
		if log (cong_pos /cong_ned )> bayesProbThresh then label is 1.
		Parameters:
			- movingWinLenIndx - length of window over witch it accumulates probabilities
			- bayesProbThresh - if accumulated probablity is higher then this threshold then it is confident enough that it is event
			- distanceBetweenEventsIndx - if two event events are closer then this, it merges them together

		'''

		if not torch.is_tensor(prediction0): #if array passed
			prediction=torch.from_numpy(prediction0)
			probability = torch.from_numpy(probability0)
		else:
			prediction=prediction0
			probability=probability0

		# convert probability to probability of pos
		probability_pos = torch.where(prediction == 0, 1 - probability, probability)

		# first classifying as true 1 if at least  GeneralParams.movingWinLen in a row is 1
		p = torch.nn.ConstantPad1d((self.movingWinLenIndx - 1, 0), 0)
		unfolded_probability = probability_pos.unfold(0, self.movingWinLenIndx, 1)
		conf_pos = unfolded_probability.prod(dim=1)
		conf_neg = (1 - unfolded_probability).prod(dim=1)
		conf = ((conf_pos + 0.00000001) / (conf_neg + 0.00000001)).log()
		out = p(torch.where(conf >= self.bayesProbThresh, 1, 0))

		# merge close events
		# second part
		smoothLabelsStep2 = torch.clone(out)
		# Calculate event starts and stops
		events = self.calculateStartsAndStops(smoothLabelsStep2)
		# if  event started but is too close to previous one, delete second event by connecting them
		for idx in range(1, len(events)):
			if events[idx][0] - events[idx - 1][1] <= self.distanceBetweenEventsIndx:
				smoothLabelsStep2[events[idx - 1][1]:events[idx][0]] = 1

		if not torch.is_tensor(prediction0):
			return (out.numpy(), smoothLabelsStep2.numpy())
		else:
			return (out, smoothLabelsStep2)


	def calculatePerformanceAfterVariousSmoothing(self, predLabels, label, probabilityLabels):
		''' Function that calculates performance for epilepsy
		It evaluates on raw predictions but also performs different postprocessing and evaluated performance after postprocessing:
			- first smoothing is just moving average with specific window size and percentage of labels that have to be 1 to give final label 1
			- then merging of too close event is performed in step2
			- another option for postprocessing and smoothing of labels is bayes postprocessing
		Returns dictionary with 9 values for each postprocessing option:
			['Sensitivity events', 'Precision events', 'F1score events', 'Sensitivity duration',
			'Precision duration', 'F1score duration', 'F1DEmean', 'F1DEgeoMean', 'numFPperDay']
		Also returns postprocessed (smoothed) labels
		'''

		performancesAll={}
		smoothedPredictions={}

		# calculate different performance measures - only no Smooth
		performancesAll['RawPred'] = self.performance_all9(predLabels, label)

		# smoothing using moving average and then merging
		(smoothedPredictions['MovAvrg'], smoothedPredictions['MovAvrg&Merge']) = self.smoothenLabels_movingAverage(predLabels)
		performancesAll['MovAvrg'] = self.performance_all9(smoothedPredictions['MovAvrg'], label)
		performancesAll['MovAvrg&Merge'] = self.performance_all9(smoothedPredictions['MovAvrg&Merge'], label)

		# bayes smoothing
		(smoothedPredictions['Bayes'], smoothedPredictions['Bayes&Merge']) = self.smoothenLabels_Bayes(predLabels, probabilityLabels)
		performancesAll['Bayes'] = self.performance_all9(smoothedPredictions['Bayes'], label)
		performancesAll['Bayes&Merge'] = self.performance_all9(smoothedPredictions['Bayes&Merge'], label)


		return (performancesAll, smoothedPredictions)


	def plotInterp_PredAndConf(self, trueLabels,predLabels, predProbability, smoothedPredictions, outputFile):
		''' function that plots in time true labels, raw predictions as well as postprocessed predictions '''
		FSize=12
		colormap = np.array([[0, 134, 139], [139, 34, 82]]) / 256 # blue then red

		YelemNames= ['TrueLabels', 'Raw']+list(smoothedPredictions.keys()) #'TrueLabels',
		numYElems = len(YelemNames)
		smoothedPredictions['Raw']=predLabels
		smoothedPredictions['TrueLabels'] = trueLabels

		fig1 = plt.figure(figsize=(10, 3), constrained_layout=False)
		gs = GridSpec(numYElems, 1, figure=fig1)
		fig1.subplots_adjust(wspace=2.2, hspace=0.2)
		xValues = np.arange(0, len(trueLabels), 1)

		for f in range(numYElems):
			ax1 = fig1.add_subplot(gs[f, 0])
			# plotting feature values
			prob = np.reshape(predProbability, (-1, 1))
			if (YelemNames[f]=='Raw'): #only if raw predictions give color transparency that marks confidence of prediction
				c1 = np.hstack((colormap[smoothedPredictions[YelemNames[f]], :], np.power(prob, 4)))
			else: #for postprocessed labels no confidence available (at the moment)
				c1= np.hstack((colormap[smoothedPredictions[YelemNames[f]], :], np.reshape(np.ones(len(trueLabels)), (-1, 1)) ))

			# plot bar plots
			barsToPlot = np.ones(len(trueLabels))
			ax1.bar(xValues, barsToPlot, color=c1, width=1.0 )
			ax1.set_ylabel(YelemNames[f], rotation=0, fontsize=FSize)
			ax1.yaxis.set_label_coords(-0.1,0)
			ax1.set_xlim([-0.5,len(trueLabels)-0.5])
			ax1.set_yticks([])
			if (f==numYElems-1):
				ax1.set_xlabel('Time [s]', fontsize=FSize)
			ax1.grid()

		fig1.show()
		fig1.savefig(outputFile + '.png', bbox_inches='tight', dpi=100)
		# fig1.savefig(outputFile+ '.svg', bbox_inches='tight')
		plt.close(fig1)

