import numpy as np
import processData

gen=processData.fetchImages(128)

print gen
count=0
i=0
while(count<=5):

	print i
	i=i+1
	n=np.random.randint(0,5)
	if n ==2:
		print ('n: %d' %(n))
		count=count+1
	# else:
		# i=i-1

