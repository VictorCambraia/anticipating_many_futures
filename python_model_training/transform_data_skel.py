import numpy as np
import cPickle
ws = 50
 


data = {}
data['x_train'] = []
data['y_train'] = []
data['x_valid'] = []
data['y_valid'] = []
data['x_test']  = []
data['y_test']  = []

order = np.array([2, 9, 5, 8, 1, 3, 0, 7, 4, 6])

data_path = ""

for number in range(1,14):
	data = {}
	data['x_train'] = []
	data['y_train'] = []
	data['x_valid'] = []
	data['y_valid'] = []
	data['x_test']  = []
	data['y_test']  = []
		
	if number == 1:
		filepath = data_path + "/21_01_2017_1130/pose_"
	if number == 2:
		filepath = data_path + "/24_01_2017_0830/pose_"
	if number == 3:
		filepath = data_path + "/27_01_2017_0900/pose_"
	if number == 4: 
		filepath = data_path + "/28_01_2017_0900/pose_"
	if number == 5:
		filepath = data_path + "/28_01_2017_1445/pose_"
	if number == 6:
		filepath = data_path + "/30_01_2017_1020/pose_"
	if number == 7:
		filepath = data_path + "/02_02_2017_0815/pose_"
	if number == 8:
		filepath = data_path + "/03_02_2017_1500/pose_"
	if number == 9:
		filepath = data_path + "/04_02_2017_1115/pose_"
	if number == 10:
		filepath = data_path + "/06_02_2017_1345/pose_"
	if number == 11:
		filepath = data_path + "/06_02_2017_1410/pose_" 
	if number == 12:
		filepath = data_path + "/06_02_2017_1420/pose_"
	if number == 13:
		filepath = data_path + "/06_02_2017_1435/pose_"
	
	
	
	savename = "data" + str(number) + ".pkl"

	for i_r in range(8):
	    i_rec = order[i_r] 
	    text_file = open(filepath +  str(i_rec) + "_u_0.txt", "r")
	    lines = text_file.read().split(' ')
	    lines = map(float, lines[:-1])
	    text_file.close() 
	    N    = int(lines[0])
	    temp_data = np.array(lines[1:]) 
	    T = temp_data.shape[0] / N
	    print "opened " + str(i_rec) + " with length " + str(T)   
	    temp_data = np.reshape(temp_data,(T, N))
	    temp_data = temp_data[1:,:]
	    temp_data = temp_data[:,3:27]
	    N    = temp_data.shape[1]
	    for t in range(0,T-2*ws):
	        if t == 0:
	            running_data_x = np.reshape(temp_data[t:t+ws],(1,ws*N))
	            running_data_y = np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))
	        else:
	            running_data_x = np.concatenate((running_data_x,np.reshape(temp_data[t:t+ws],(1,ws*N))))
	            running_data_y = np.concatenate((running_data_y,np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))))
	    print "append " + str(i_rec) + " with length " + str(running_data_x.shape[0])   
	    if i_r == 0:
	        data['x_train'] = running_data_x
	        data['y_train'] = running_data_y
	    else:
	        data['x_train'] = np.concatenate((data['x_train'],running_data_x))
	        data['y_train'] = np.concatenate((data['y_train'],running_data_y))
	        
	 
	
	data['x_train'] = data['x_train'] 
	data['y_train'] = data['y_train'] 
	
	
	for i_r in [8]:
	    i_rec = order[i_r] 
	    text_file = open(filepath+  str(i_rec) + "_u_0.txt", "r")
	    lines = text_file.read().split(' ')
	    lines = map(float, lines[:-1])
	    text_file.close() 
	    N    = int(lines[0])
	    temp_data = np.array(lines[1:]) 
	    T = temp_data.shape[0] / N
	    print "opened " + str(i_rec) + " with length " + str(T)   
	    temp_data = np.reshape(temp_data,(T, N))
	    temp_data = temp_data[1:,:]
	    temp_data = temp_data[:,3:27]
	    N    = temp_data.shape[1]
	    for t in range(0,T-2*ws):
	        if t == 0:
	            running_data_x = np.reshape(temp_data[t:t+ws],(1,ws*N))
	            running_data_y = np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))
	        else:
	            running_data_x = np.concatenate((running_data_x,np.reshape(temp_data[t:t+ws],(1,ws*N))))
	            running_data_y = np.concatenate((running_data_y,np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))))
	    print "append " + str(i_rec) + " with length " + str(running_data_x.shape[0])   
	    if i_r == 8:
	        data['x_valid'] = running_data_x
	        data['y_valid'] = running_data_y
	    else:
	        data['x_valid'] = np.concatenate((data['x_valid'],running_data_x))
	        data['y_valid'] = np.concatenate((data['y_valid'],running_data_y))
	        
	T_now = data['x_valid'].shape[0]
	
	 
	
	data['x_valid'] = data['x_valid'] 
	data['y_valid'] = data['y_valid'] 
	
	for i_r in [9]:
	    i_rec = order[i_r] 
	    text_file = open(filepath +  str(i_rec) + "_u_0.txt", "r")
	    lines = text_file.read().split(' ')
	    lines = map(float, lines[:-1])
	    text_file.close() 
	    N    = int(lines[0])
	    temp_data = np.array(lines[1:]) 
	    T = temp_data.shape[0] / N
	    print "opened " + str(i_rec) + " with length " + str(T)   
	    temp_data = np.reshape(temp_data,(T, N))
	    temp_data = temp_data[1:,:]
	    temp_data = temp_data[:,3:27]
	    N    = temp_data.shape[1]
	    for t in range(0,T-2*ws):
	        if t == 0:
	            running_data_x = np.reshape(temp_data[t:t+ws],(1,ws*N))
	            running_data_y = np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))
	        else:
	            running_data_x = np.concatenate((running_data_x,np.reshape(temp_data[t:t+ws],(1,ws*N))))
	            running_data_y = np.concatenate((running_data_y,np.reshape(temp_data[t+ws:t+2*ws],(1,ws*N))))
	    print "append " + str(i_rec) + " with length " + str(running_data_x.shape[0])   
	    if i_r == 9:
	        data['x_test'] = running_data_x
	        data['y_test'] = running_data_y
	    else:
	        data['x_test'] = np.concatenate((data['x_test'],running_data_x))
	        data['y_test'] = np.concatenate((data['y_test'],running_data_y))
	        
	 
	
	data['x_test'] = data['x_test'] 
	data['y_test'] = data['y_test'] 
	
	
	f = open(data_path+savename, 'wb')
	cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	print "saved " + data_path+savename
	
	
	
	
	    
