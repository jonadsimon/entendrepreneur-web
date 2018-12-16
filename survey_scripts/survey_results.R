df <- read.csv('~/code/what-do-you-call-a-bot/data/comparison_surveys/survey_data.csv')
df <- df[df$Answer != '',]
df_summary <- table(df[,3:5])
dim(df_summary)
df_summary

# reshape 
# for each 3,4 grouping, get counts of 5

# Perform binomial significance tests

binom.test(34,21+34,0.5,alternative="greater") # Charmanteau better puns: p=0.052 , n=55
binom.test(28,26+28,0.5,alternative="greater") # Port Manteaux better puns: p=0.446, n=54

binom.test(47,47+26,0.5,alternative="greater") # Charmanteau less funny: p=0.009, n=73
binom.test(30,30+17,0.5,alternative="greater") # Port Manteaux less funny: p=0.039, n=47

binom.test(34,21+34,0.5,alternative="two.sided") # Charmanteau better puns: p=0.10 , n=55
binom.test(28,26+28,0.5,alternative="two.sided") # Port Manteaux better puns: p=0.446, n=54

binom.test(47,47+26,0.5,alternative="two.sided") # Charmanteau less funny: p=0.009, n=73
binom.test(30,30+17,0.5,alternative="two.sided") # Port Manteaux less funny: p=0.079, n=47
