address_out=['0x60640018','0x606976fc','0x606eede0','0x607464c4','0x6079dba8','0x607f528c',
		     '0x6084c970','0x608a4054','0x608fb738','0x60952e1c','0x609aa500','0x60a01be4',
			'0x60a592c8','0x60ab09ac','0x60b08090','0x60b5f774']

address_in=['0x6066b130','0x606c2814','0x60719ef8','0x607715dc','0x607c8cc0','0x608203a4',
		     '0x60877a88','0x608cf16c','0x60926850','0x6097df34','0x609d5618','0x60a2ccfc',
			'0x60a843e0','0x60adbac4','0x60b331a8','0x60b8a88c']

profile_out=['0x60696248','0x606ed92c','0x60745010','0x6079c6f4','0x607f3dd8','0x6084b4bc',
		'0x608a2ba0','0x608fa284','0x60951968','0x609a904c','0x60a00730','0x60a57e14',
			'0x60aaf4f8','0x60b06bdc','0x60b5e2c0','0x60bb59a4']

chip_list=['0 0','1 0','1 1','0 1','2 0','2 1','2 2','1 2','0 2','3 0','3 1','3 2','4 3','3 3','2 3','1 3','4 0','4 1','4 2','5 3','5 4','4 4','3 4','2 4','0 3','5 1','5 2','6 3','6 4','6 5','6 6','5 5','4 6','4 5','3 5','1 4','6 2','7 3','7 4','7 5','7 6','7 7','6 7','5 6','5 7','4 7','3 6','2 5']

#print len(chip_list)

def input_gen(chips=1,segments=1,cores=16):

	f=open('./timed_input_write','w')
	#f.write('test.\n')
	for j in range(segments):
		for k in range(chips):
			f.write("sp {}\n".format(chip_list[k]))
			for i in range(cores):
				line="sload ./load_files/load{}_{} {}\n".format((cores*k)+i+1,j+1,address_in[i])
				f.write(line)
		if j==0:	
			f.write("app_sig all 20 sync0\n")
	#		f.write("sleep 0.5\n")
		else:
			f.write("sleep\n")
	f.close()

def dump(chips,segment_length,num_seg,num_fibres=10,cores=16):



	bytes=segment_length*num_seg*num_fibres*4 
	profile_bytes=3*num_seg*4
	f=open('./RAM_dump','w')
	for k in range(chips):
		f.write("sp {}\n".format(chip_list[k]))
		for i in range(cores):	
			line="sdump ./dump_files/dump{}.bin {} {}\n".format((cores*k)+i+1,address_out[i],hex(bytes))
			f.write(line)
			#profile dump
			f.write("sdump ./dump_files/profile{}.bin {} {}\n".format((cores*k)+i+1,profile_out[i],hex(profile_bytes)))	
	f.close()
		
def test_script_gen(segments,chips=1,dur=7,boot_string="boot",cores=16,segment_length=100,num_fibres=10,app_no=20):

	#generate up to date timed_input_write file
	input_gen(chips,1,cores)#1 segment inputs hard coded
	#generate up to date dump file
	dump(chips,segment_length,segments,num_fibres,cores)

	f=open('./test','w')
#	f.write("boot scamp.boot no_wdog.conf\n")	
	f.write("{}\n".format(boot_string))
	f.write("app_stop {}\n".format(app_no))	
	for i in range(chips):
		#switch chip
		f.write("sp {}\n".format(chip_list[i]))
		#load application 
		f.write("@ application_load_single_chip\n")
	f.write("@ timed_input_write\n")
	f.write("sleep {}\n".format(dur))	
	f.write("@ RAM_dump\n")


test_script_gen(segments=200,chips=1,dur=5,boot_string="boot scamp.boot no_wdog.conf",cores=16,segment_length=100,num_fibres=1,app_no=20)


