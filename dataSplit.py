import random

folder = "data/"
# f = open(folder + "swm_input_data.csv")
# f = open(folder + "swm_testing.csv")
f = open(folder + "swm_training.csv")
w_training_pos = open(folder + "swm_training_pos.csv","w")
w_training_neg = open(folder + "swm_training_neg.csv","w")
for line in f:
    line = line.strip()
    if not line: 
        continue
    email, tag = line.split(' ')
    if int(tag) == 1:
        w_training_pos.write(email+" "+str(tag)+"\n")
    if int(tag) == 0:
        w_training_neg.write(email+" "+str(tag)+"\n")

'''
w_testing = open(folder + "swm_testing.csv", "w")
w_validation = open(folder + "swm_validation.csv", "w")
# i = 0
# isPos = True
# for line in f:
#     line = line.strip()
#     if not line: 
#         continue
#     email, tag = line.split(' ')
#     if i>=3000:
#         break
#     if int(tag) == 1 and isPos:
#         w_validation.write(email+" "+str(tag)+"\n")
#         isPos = False
#     if int(tag) == 0 and isPos == False:
#         w_validation.write(email+" "+str(tag)+"\n")
#         isPos = True
#         i+=1




training_1 = 0
testing_1 = 0

training_total = 0
testing_total = 0

splitRate = 0.7
max_validation = 2000
isPos= True

for line in f:
    line = line.strip()
    email, tag = line.split('\x07')
    if random.random() <= splitRate:
        w_training.write(email+" "+str(tag)+"\n")
        if int(tag) == 1:
            w_training_pos.write(email+" "+str(tag)+"\n")
        else:
            w_training_neg.write(email+" "+str(tag)+"\n")
        training_1 += int(tag)
        training_total += 1
    else:
        w_testing.write(email+" "+str(tag)+"\n")
        testing_1 += int(tag)
        testing_total += 1
        if max_validation:
            if isPos and int(tag):
                w_validation.write(email+" "+str(tag)+"\n")
                isPos = False
            elif isPos == False and int(tag) == 0:
                w_validation.write(email+" "+str(tag)+"\n")
                isPos = True
                max_validation -= 1
 
print("training len: %d, training_1 percentage: %f"%(training_total, float(training_1)/training_total))
print("testing len: %d, testing_1 percentage: %f"%(testing_total, float(testing_1)/testing_total))

'''