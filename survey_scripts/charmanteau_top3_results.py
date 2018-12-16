# SFile should be in Charmanteau-CamReady/Code
import barebones_enc_dec as bed

# load random word-pairs to try
input_file = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/random_word_pairs.txt'
with open(input_file) as infile:
	input_pairs = [line.strip().split() for line in infile.readlines()]

max_pairs = 200
input_pairs = input_pairs[:max_pairs]

# Load the model
predictor = bed.getModel()

print 'Finished Loading Model'

output_file = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/charmanteau_top3_results.txt'

with open(output_file, 'w') as outfile:
	for i, (g1,g2) in enumerate(input_pairs):
		top3 = bed.query(g1,g2,predictor)[:3]

		outfile.write('{},{}'.format(g1,g2))
		for p in top3:
			outfile.write(',{}'.format(p))
		outfile.write('\n')

		print 'Finished {} pairs, {} remaining'.format(i+1, max_pairs-i-1)