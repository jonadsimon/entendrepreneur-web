# browser is running javascript which pings an API... easier to just copy/paste all relevant urls and download results
# dump all urls of interest to a file, to be manually entered into the chrome browser

import glob
import os
from bs4 import BeautifulSoup
import re

def get_url(word1,word2):
	return 'https://www.onelook.com/pm/#?w1={}&w2={}'.format(word1,word2)

def get_url_bash_calls():
	# load random word-pairs to try
	input_file = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/random_word_pairs2.txt'
	with open(input_file) as infile:
		input_pairs = [line.strip().split() for line in infile.readlines()]

	max_pairs = 200
	input_pairs = input_pairs[:max_pairs]

	# print commands to type into terminal for generating associated Chrome tabs (10 at a time)
	c = 0
	for i in range(0,200,10):
		url_list = [get_url(input_pairs[j][0], input_pairs[j][1]) for j in range(i,i+10)]
		print "/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --new-window --new-tab '{}'\n\n".format("' '".join(url_list))

def convert_html_dumps_to_top3():
	dir_path = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/portmanteaux_results/'
	result_file = '/Users/jonathansimon/code/what-do-you-call-a-bot/data/portmanteaux_top3_results.txt'
	files = glob.glob('{}*.html'.format(dir_path))
	files.sort(key=os.path.getmtime)
	with open(result_file, 'w') as outfile:
		for filename in files:
			with open(filename) as infile:
				html = ''.join([line for line in infile.readlines()])
			parsed_html = BeautifulSoup(html)

			# capture the input words
			title_string = parsed_html.find('title').text
			matches = re.match('Port Manteaux: (.*) mixed with (.*)', title_string)
			word1 = matches.group(1)
			word2 = matches.group(2)
			outfile.write('{},{}'.format(word1,word2)) # write to the head of the row

			# extract the top-3 results from each row
			table_rows = parsed_html.body.find('div', attrs={'class':'results_div'}).find_all('tr')
			# skip the first header row
			for i in range(1,4):
				row = table_rows[i].find_all('td')
				outfile.write(',{}'.format(row[5].text))
			# write a newline break
			outfile.write('\n')

# get_url_bash_calls() # UNCOMMENT TO GENERATE TERMINAL COMMANDS
convert_html_dumps_to_top3() # UNCOMMENT TO GENERATE PARSE AND DUMP TOP 3 RESULTS
