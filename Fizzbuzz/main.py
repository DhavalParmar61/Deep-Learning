import tensorflow as tf
import numpy as np
import sys
import os

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizzbuzz(i,prediction):
    return[str(i),"fizz","buzz","fizzbuzz"][prediction]

NUM_DIGITS = 10
test_file = sys.argv[2]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(150,activation='relu'),
    tf.keras.layers.Dense(150,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')]
)
trained_model = tf.keras.models.load_model("./model/trained_model.h5")

file = open(test_file,'r')
test_input = []
for line in file:
    line = line.replace('\n','')
    test_input.append(line)

test_input = [int(i) for i in test_input]
test_input_b = np.array([binary_encode(i, NUM_DIGITS) for i in test_input])
predictions = trained_model.predict_classes(test_input_b)

repo_path = os.path.dirname(os.path.abspath(__file__))
out_sw_2_fp = repo_path + "/Software2.txt"
file = open("Software2.txt","w")
for i in range(len(test_input)):
    file.write(fizzbuzz(test_input[i],int(predictions[i]))+"\n")
file.close()

'''%%%%%%%%---------------------------------------------------
-----------------------------------------------------------
---------------------------------------------------%%%%%%%%'''

out_sw_1_fp = repo_path + "/Software1.txt"
file1 = open("Software1.txt","w")
for i in range(len(test_input)):
    if (test_input[i]%15 == 0):
        file1.write("fizzbuzz\n")
    elif(test_input[i]%5 == 0):
        file1.write("buzz\n")
    elif(test_input[i]%3 == 0):
        file1.write("fizz\n")
    else:
        file1.write(str(test_input[i])+"\n")
file1.close()

f1=open("Software1.txt","r")
f2=open("Software2.txt","r")
a=[]
b=[]
for line in f1:
    a.append(line)

for line in f2:
    b.append(line)
print(len(b))
count=0
for i in range(len(a)):
    if(a[i]==b[i]):
        count=count+1
print(count)