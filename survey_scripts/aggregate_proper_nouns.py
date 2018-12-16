# cities_filename = 'data/names_locations/cities15000.txt' # too many city names that are also real words
states_filename = 'data/names_locations/admin1CodesASCII.txt'
countries_filename = 'data/names_locations/countryInfo.txt'
continents_filename = 'data/names_locations/continents.txt'
first_names_filename = 'data/names_locations/census_first_names.txt' # only use first 1000
last_names_filename = 'data/names_locations/census_last_names.txt' # only use first 1000

# assume all tab-separated
# with open(cities_filename) as infile:
# 	cities = [line.strip().split('\t')[1].lower().replace(' ', '_') for line in infile.readlines()]

# print 'Finished loading cities'

# assume all tab-separated
with open(states_filename) as infile:
	states = [line.strip().split('\t')[1].lower().replace(' ', '_') for line in infile.readlines()]

print 'Finished loading states'

# assume all tab-separated
with open(countries_filename) as infile:
	countries = [line.strip().split('\t')[4].lower().replace(' ', '_') for line in infile.readlines()]
	cities = [line.strip().split('\t')[5].lower().replace(' ', '_') for line in infile.readlines()]

print 'Finished loading countries'
print 'Finished loading cities'

with open(continents_filename) as infile:
	continents = [line.strip().lower().replace(' ', '_') for line in infile.readlines()]

print 'Finished loading continents'

# assume all tab-separated
with open(first_names_filename) as infile:
	first_names = [line.strip().split()[0].lower() for line in infile.readlines()]
	first_names = first_names[:1000] # only take most common 1000, too many "real" words mixed in 

print 'Finished loading first names'

# assume all tab-separated
with open(last_names_filename) as infile:
	last_names = [line.strip().split()[0].lower() for line in infile.readlines()]
	last_names = last_names[:1000] # only take most common 1000, too many "real" words mixed in 

print 'Finished loading last names'

# dump ALL of these into a single "proper nouns" file
outfile_path = 'data/names_locations/common_proper_nouns.txt'
with open(outfile_path, 'w') as outfile:
	for w in cities:
		outfile.write(w+'\n')
	print 'Finished dumping cities'

	for w in states:
		outfile.write(w+'\n')
	print 'Finished dumping states'

	for w in countries:
		outfile.write(w+'\n')
	print 'Finished dumping countries'

	for w in continents:
		outfile.write(w+'\n')
	print 'Finished dumping continents'

	for w in first_names:
		outfile.write(w+'\n')
	print 'Finished dumping first names'

	for w in last_names:
		outfile.write(w+'\n')
	print 'Finished dumping last names'