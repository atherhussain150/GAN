def group(arr):
    counter = {}
    for i in arr:
        if i in counter:
            counter[i] +=1
        else:
            counter[i] = 1
    result = [[key,value] for key,value in counter.items()]
    result = sorted(result, key=lambda x: (-x[1], x[0]))
    print(result)
    
list = [2,4,4,6,2,1,1]
group(list)
arra = [1,2,3,4,5,6]
print(-arra[0])