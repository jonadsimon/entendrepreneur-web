import random

random.seed(0)

data_dir = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/'
entrendrepeneur_file_path = data_dir+'entendrepeneur_top3_results.txt'
portmanteaux_file_path = data_dir+'portmanteaux_top3_results.txt'
charmanteau_file_path = data_dir+'charmanteau_top3_results.txt'

with open(entrendrepeneur_file_path) as infile:
	entendrepeneur_top_3s = [line.strip().split(',') for line in infile.readlines()]

with open(portmanteaux_file_path) as infile:
	portmanteaux_top_3s = [line.strip().split(',') for line in infile.readlines()]

with open(charmanteau_file_path) as infile:
	charmanteau_top_3s = [line.strip().split(',') for line in infile.readlines()]

# survey, question, choice, word1, word2, portmanteau, algorithm
competitors = ['portmanteaux','charmanteau']
for i in range(0,200,10):
	with open(data_dir+'comparison_surveys/questionnaires/survey{}.txt'.format(i/10 + 1), 'w') as outfile: # UNCOMMENT TO GENERATE QUESTIONS
	# with open(data_dir+'comparison_surveys/solutions/survey{}_sol.txt'.format(i/10 + 1), 'w') as outfile: # UNCOMMENT TO GENERATE ANSWER KEY
		for j in range(i,i+10):
			# pick a competitor
			competitor = random.sample(competitors,1)[0]
			if competitor == 'portmanteaux':
				competitor_row = portmanteaux_top_3s[j]
			else:
				competitor_row = charmanteau_top_3s[j]
			# sanity check
			if competitor_row[0] != entendrepeneur_top_3s[j][0] or competitor_row[1] != entendrepeneur_top_3s[j][1]:
				print j
				print competitor_row[0], entendrepeneur_top_3s[j][0]
				print competitor_row[1], entendrepeneur_top_3s[j][1]
				raise Exception('WORD PAIRS DO NOT MATCH!!!')

			# print a line break to demarcate the two question types
			if j % 10 == 5:
				outfile.write('\n\n\n'+'-'*80)

			# need to track which are mine before shuffling
			shuffled_solutions = random.sample(zip(entendrepeneur_top_3s[j][2:5],['entendrepeneur']*3) + zip(competitor_row[2:5],[competitor]*3), 6)
			outfile.write('\n\n\nQuestion {}: {}, {}\n\n'.format(j-i+1, competitor_row[0], competitor_row[1])) # UNCOMMENT TO GENERATE QUESTIONS
			# outfile.write('\n\n\nQuestion {}: {}, {} ({})\n\n'.format(j-i+1, competitor_row[0], competitor_row[1], competitor)) # UNCOMMENT TO GENERATE ANSWER KEY

			for k in range(6):
				outfile.write('{}\n'.format(shuffled_solutions[k][0])) # UNCOMMENT TO GENERATE QUESTIONS
				# outfile.write('{}. {} ({})\n'.format(chr(97+k), shuffled_solutions[k][0], shuffled_solutions[k][1]))
			outfile.write('Other\n')