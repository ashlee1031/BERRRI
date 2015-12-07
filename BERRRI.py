#!/usr/bin/python

import getopt,sys,os,warnings,operator,re,subprocess,copy
import numpy as np
import scipy
import scipy.special
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import csv
import math
from collections import OrderedDict
import datetime
from matplotlib.font_manager import FontProperties
from random import shuffle
import warnings
warnings.filterwarnings('error')


######################## MISC FUNCTIONS ##############################

# Function to compute the intersection of 2 lists, and return 
# shared elements and indices of those in each list
def intersect(list1,list2):
	# Init Empty Dictionaries
	d1 = {}; 
	d2 = {};
	#Turn each list into a dictionary, where the elements are the keys and the indices are the values
	for i in range(len(list1)): 
		d1[list1[i]] = i;
	for i in range(len(list2)):
		d2[list2[i]] = i;
	#Cast the arrays to lists so we can concatenate them
	if (type(list1).__module__ =='numpy'): 
		list1 = list1.tolist();
		list2 = list2.tolist();
	combined = list1+list2;
	#Get the unique elements
	uniqueelems = OrderedDict.fromkeys(combined).keys();
	sharedelems = [];
	#Figure out which elements are shared
	for elem in uniqueelems:
		if (elem in d1.keys() and elem in d2.keys()):
			sharedelems.append(elem);
	#Figure out the indices of those shared elements
	ind1 = [];
	ind2 = [];
	for elem in sharedelems:
		if (elem in d1.keys()):
			ind1.append(d1[elem]);
		if (elem in d2.keys()):
			ind2.append(d2[elem]);
	return sharedelems,ind1,ind2
 
# Function to determine whether a string is a valid number  
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
		
# Function to generate a readable time elapsed string
def delta_time_string(prevtime,curtime):
	hrs = curtime.hour-prevtime.hour;
	mins = curtime.minute-prevtime.minute;
	seconds = curtime.second-prevtime.second;
	miseconds = curtime.microsecond-prevtime.microsecond;
	
	if (mins < 0):
		hrs = hrs - 1;
		mins = 60 + mins;
	if (seconds < 0):
		mins = mins - 1;
		seconds = 60 + seconds;
	if (miseconds < 0):
		seconds = seconds - 1;
		miseconds = 1000000 + miseconds;
			
	return str(hrs)+" hours, "+str(mins)+" minutes, "+str(seconds)+" seconds, and "+str(miseconds)+" microseconds"

#Function to Compute the Residual Sum Of Squares	
def computeRSS(y1,y2):
	return ((y1-y2)**2).sum();

#Function to Compute the Variation	
def computeCV(y1,y2):
	return scipy.stats.variation(np.reshape(y1-y2,(np.shape(y1)[0]*np.shape(y1)[1])))

#Split the Data for Cross-Validation	
def splitXYcrossval(x,y,npart):
	
	splitx = {};
	splity = {};
	# Generate random partitions
	nsamples = np.shape(x)[0];
	indices = range(nsamples);
	shuffle(indices);
	
	full_true_y = [ y[i,:] for i in indices ];
		
	num_in_each = int(nsamples/npart);
	
	start = 0;
	for i in range(npart):
		#Get the x and y data
		thisx = x[start:start+num_in_each,:];
		thisy = y[start:start+num_in_each,:];
		start = start+num_in_each;
		#If there are at least "num_in_each" samples left, write out the files
		if (nsamples >= start+num_in_each):
			splitx[i] = thisx;
			splity[i] = thisy
		#Otherwise, add the rest then write out the file
		else:
			if (start < nsamples):
				thisx = np.concatenate((thisx,x[start:,:]),axis=0);
				thisy = np.concatenate((thisy,y[start:,:]),axis=0);
			splitx[i] = thisx;
			splity[i] = thisy
	
	# Return the split x and y file names
	return splitx,splity,full_true_y  


#################################################################################
	
	
######################## SNPTEST READING FUNCTIONS ##############################

# Function to read in the genotypes in SNPTEST format
def readSNPTESTGenotypes(genofile,maflimit,outdir):
	print "Reading Genotypes...";
	orig_geno = genofile;
	if (genofile.endswith('.gz')):
		os.system("cp "+genofile+" "+outdir)
		genofile = outdir+genofile.split('/')[len(genofile.split('/'))-1]
		os.system("gunzip -f "+ genofile);
		genofile = genofile.replace('.gz','');
	nsnps = 0;
	snpid = [];
	rsid = [];
	snploc = [];
	al1 = [];
	al2 = [];
	maf = [];
	cols = [];
	ncols = 0;
	genos = [];
	counter = 0;
	BUFFER = int(1.6E10) #2 gb buffer
	file = open(genofile, 'r')
	text = file.readlines(BUFFER)
	while text != []:
		for t in text:
			line = t.rstrip('\n');
			if line != "":
				if nsnps == 0:
					cols = line.split(' ');
					cols = filter(None, cols);
					ncols = len(cols);
					nind = (ncols-5)/3;
					genos = np.zeros((1,nind));
				cols = cols = line.split(' ');
				snpid.append(cols[0]);
				rsid.append(cols[1]);
				snploc.append(int(cols[2]));
				al1.append(cols[3]);
				al2.append(cols[4]);
				counter = 5;
				if (nsnps > 0):
					genos = np.concatenate((genos,np.zeros((1,nind))));

				for j in range(nind):
					temp = cols[counter:counter+3];
					if (len(temp)>0):
						ind = np.array(temp,dtype=float).argmax(axis=0); 
						genos[nsnps,j] = ind;
						counter = counter + 3;
				nsnps = nsnps + 1;
				if ((nsnps) % 1000 == 0):
					print "Finished reading "+str(nsnps)+" SNPs..."
					sys.stdout.flush();
		text = file.readlines(BUFFER)

	file.close();

	maf = np.array(np.sum(genos,1)/(nind*2),dtype=float);
	snpid = np.array(snpid,dtype=str);
	rsid = np.array(rsid,dtype=str);
	al1 = np.array(al1,dtype=str);
	al2 = np.array(al2,dtype=str);
	snploc = np.array(snploc,dtype=int);

	if (maflimit > 0.5):
		maflimit = 1-maflimit;
	mask = np.ones(maf.shape,dtype=bool)
	mask[maf < maflimit] = 0;
	mask[maf > 1-maflimit]=0;

	genos = genos[mask,:];
	maf = maf[mask];
	al1 = al1[mask];
	al2 = al2[mask];
	snpid = snpid[mask];
	rsid = rsid[mask];
	snploc = snploc[mask];
	print str(len(mask))+" SNPs before MAF Filtering..."
	print str(len(maf))+" SNPs after MAF Filtering..."
	samplefile = orig_geno.replace('.gens','.samples')
	samplefile = samplefile.replace('.phased.impute2.gen','.samples')
	samplefile = samplefile.replace('.gz','')
	fh = open(samplefile,'r');
	lines = fh.readlines();
	fh.close();
	sampleids = [];
	for line in lines:
		cols = line.split(' ');
		sampleids.append(cols[0]);

	sys.stdout.flush()
	return genos.T,maf,al1,al2,snpid,rsid,snploc,sampleids;

# Function to read in the gene expression in SNPTEST format
def readSNPTESTGeneXpr(xprfile,info_file,chrom):

	print "Reading Gene Expression...";
	
	fh = open(xprfile,'r');
	lines = fh.readlines();
	fh.close();
	lines = filter(None, lines);
	genes = lines[0].split(' ');
	genes = genes[3:];
	ngenes = len(genes);
	lines = lines[2:];
	nsamples = len(lines);
	sampleids = [];
	xpr = np.zeros((ngenes,nsamples));

	for i in range(nsamples):
		lines[i] = lines[i].rstrip('\n');
		cols = cols = lines[i].split(' ');
		sampleids.append(cols[0]);
		xpr[:,i] = np.array(cols[3:],dtype=float).T;
	
	fh = open(info_file,'r');
	lines = fh.readlines();
	fh.close();
	
	genesinfo = [];
	trans = [];
	chroms = [];
	loc1 = [];
	loc2 = [];
	
	for line in lines:
		line = line.rstrip('\n');
		cols = cols = line.split('\t');
		genesinfo.append(cols[2]);
		trans.append(cols[3]);
		chroms.append(cols[4]);
		loc1.append(cols[5]);
		loc2.append(cols[6]);
		
	genes = np.array(genes,dtype=str);
	genesinfo = np.array(genesinfo,dtype=str);
	trans = np.array(trans,dtype=str);
	chroms = np.array(chroms,dtype=str);
	loc1 = np.array(loc1,dtype=int);
	loc2 = np.array(loc2,dtype=int);
	
	mask = np.ones(genesinfo.shape,dtype=bool)
	mask[chroms != chrom]=0;
	genesinfo = genesinfo[mask];
	trans = trans[mask];
	chroms = chroms[mask];
	loc1 = loc1[mask];
	loc2 = loc2[mask];
	shared,ind1,ind2 = intersect(genes,genesinfo);
	xpr = xpr[ind1,:];
	print str(len(ind1))+" genes on "+chrom+"..."
	
	sys.stdout.flush()
	return genes[ind1],sampleids,xpr.T,genesinfo[ind2],trans[ind2],chroms[ind2],loc1[ind2],loc2[ind2];

##################################################################################

######################## PARAMETER UPDATE FUNCTIONS ##############################

# Function to update the parameters on the inverse gamma prior for sigma_a
def update_sigma_a(am,av,c,d,k_limit,p):
	kappa1 = c + ((1)/2);
	kappa2 = np.zeros(np.shape(am));
	sum1 = 0;
	for i in range(k_limit):
		for p in range(np.shape(am)[1]):
			kappa2[i,p] = d + ((av[i,p] + np.dot(am[i,p],am[i,p].T))/2);
	return kappa1,kappa2

def update_all_zqk(vn, knownx, knowny, zm, am, av, n, p, q, k_limit, xxt_q, dgtaus, exp_phiphi, threshold):
    
    all_k = np.arange(0,k_limit);
    all_q = np.arange(0,q);
    new_nus = zm;
    x_xnq = np.zeros((q,q-1));
    for q1 in range(q):
        mask = np.ones(all_q.shape,dtype=bool)
        mask[q1]=0;
        not_q = all_q[mask];
        x_xnq[q1,:] = np.dot(knownx[:,q1],knownx[:,not_q]);
    for k1 in range(k_limit):
        #Get the indices for not k
        mask = np.ones(all_k.shape,dtype=bool)
        mask[k1]=0;
        not_k = all_k[mask];
                #If all Ak. are zero (i.e. if this factor affects no genes), there is no reason to update Z.k
        if (all(abs(phi_mean[k1,:]) < threshold)):
            nu[:,k1] = 0;
        else:
            resids = np.dot((knowny-np.dot(np.dot(knownx,new_nus[:,not_k]),am[not_k,:])),am[k1,:]);
            for q1 in range(q):
                mask = np.ones(all_q.shape,dtype=bool)
                mask[q1]=0;
                not_q = all_q[mask];
                sum1 = exp_phiphi[k1]*xxt_q[q1];
                sum2 = exp_phiphi[k1]*np.sum(np.dot(x_xnq[q1,:],new_nus[not_q,k1]))
                sum3 = np.sum(np.dot(knownx[:,q1],resids));
                zeta = dgtaus[k1] - (1/(2*vn))*sum1 - (1/vn)*sum2 + (1/vn)*sum3;
                if (zeta < -50):
                    new_nus[q1,k1] = 0;
                elif (zeta > 50):
                    new_nus[q1,k1] = 1;
                else:
                    new_nus[q1,k1] = (1/(1+(math.exp(-zeta))));
        if (np.all(new_nus == 0)):
            new_nus = np.random.uniform(0, 0.1, (int(q),int(k_limit)));
    return new_nus;

# Function to update the parameters on Akp when Akp has an L1 penalty
def update_all_ak_va(kappa1, kappa2, vn, knownx, knowny, am, zm, k_limit, n, p, q, xxt_qs, threshold, va, learn_vara):
    
    phi_means = am;
    phi_vars = np.zeros((k_limit,p));
    sumvs_k = np.zeros((k_limit));
    all_k = np.arange(0,k_limit);
    all_q = np.arange(0,q);
    
    for q1 in range(q):
        mask = np.ones(all_q.shape,dtype=bool)
        mask[q1]=0;
        notqi = all_q[mask];
        mask[q1]=1;
        sumvs_k = sumvs_k + xxt_qs[q1]*zm[q1,:];
        for k1 in range(k_limit):
            sumvs_k[k1] = sumvs_k[k1] + np.sum(np.dot(zm[q1,k1]*knownx[:,q1],np.dot(knownx[:,notqi],zm[notqi,k1])));

    # Update A
    for k1 in range(k_limit):
        #Get the indices for not k
        mask = np.ones(all_k.shape,dtype=bool)
        mask[k1]=0;
        not_k = all_k[mask];
        #If all Z.k are zero (i.e. if no SNP is in this factor), there is no reason to update Ak.
        if (all(nu[:,k1] < threshold)):
            phi_means[k1,:] = 0;
            phi_vars[k1,:] = 0;
        elif (learn_vara):
            sump = np.sum(np.multiply(np.tile(np.dot(knownx,zm[:,k1]),(p,1)).T,(knowny - np.dot(np.dot(knownx,zm[:,not_k]),phi_means[not_k,:]))),0);
            phi_vars[k1,:] = np.power(((kappa1/kappa2[k1,:]) + (sumvs_k[k1]/vn)),-1);
            phi_means[k1,:] = (sump/vn)*phi_vars[k1];
        else:
            sump = np.sum(np.multiply(np.tile(np.dot(knownx,zm[:,k1]),(p,1)).T,(knowny - np.dot(np.dot(knownx,zm[:,not_k]),phi_means[not_k,:]))),0);
            phi_vars[k1,:] = math.pow(((1/va) + (sumvs_k[k1]/vn)),-1);
            phi_means[k1,:] = (sump/vn)*phi_vars[k1];
    
    if (np.all(phi_means == 0)):
        phi_means = np.random.normal(0,1,(int(k_limit),int(p)));
        phi_vars = np.multiply(np.ones((int(k_limit),p)),va)
    return phi_means,phi_vars;

# Function to test for convergence
def geweke(trace,alpha):
	ntrace = np.shape(trace)[-1];
	#There has to be at least 10 iterations to even bother checking
	if (ntrace < 10):
		return False;
	else:
		seta = np.zeros(ntrace,dtype=bool);
		setb = np.zeros(ntrace,dtype=bool);
		seta[0:int(ntrace/10)] = True;
		setb[int(ntrace/2):ntrace] = True;
		try:
			z = (np.mean(trace[seta])-np.mean(trace[setb]))/np.sqrt((np.std(trace[seta])/sum(seta)) + (np.std(trace[setb])/sum(setb)));
		except (ZeroDivisionError,RuntimeWarning):
			z = np.nan;
		if (math.isnan(z) and (np.mean(trace[seta])-np.mean(trace[setb]))==0):
			z = 0;
		pval = scipy.stats.norm.sf(abs(z))*2;
		return pval > alpha;		

def geweke2(trace,alpha):
	for i in range(np.shape(trace)[0]):
		c = geweke(trace[i,:],alpha);
		if (c == False):
			return False;
			break;
	return True;

def geweke3(trace,alpha):
	for i in range(np.shape(trace)[0]):
		for j in range(np.shape(trace)[1]):
			c = geweke(trace[i,j,:],alpha);
			# print i,j,c
			if (c == False):
				return False;
				break;
	return True;
		
#Function to Run Variational Inference
def fitModel(iterations,burn,x,y,p,q,k_limit,c,d,phi_mean,nu,alpha,var_a,var_n,learn_a,learn_z,learn_avar, threshold,output_dir):
    
	#Initialize Parameters (A and Z might already be initialized so they are passed in as args)
	p = np.shape(phi_mean)[1];
	phi_var = np.multiply(np.ones((int(k_limit),p)),var_a);
	tau_k1 = np.multiply(np.ones((int(k_limit))),0.5);
	tau_k2 = np.multiply(np.ones((int(k_limit))),0.5);
	kappa1 = c + (1/2);
	kappa2 = d*np.ones((int(k_limit),p));
	n = np.shape(x)[0];

	#Store the traces
	phi_mean_traces = np.zeros((int(k_limit),int(p),iterations));
	phi_var_traces = np.zeros((int(k_limit),int(p),iterations));
	nu_traces = np.zeros((int(q),int(k_limit),iterations));
	tau_traces1 = np.zeros((int(k_limit),iterations));
	tau_traces2 = np.zeros((int(k_limit),iterations));
	kappas = np.zeros((int(k_limit),int(p),iterations));
	
	#Compute some things at various steps to try and speed things up a bit
	exp_phiphis = np.zeros((int(k_limit))); #Expectation of Ak. Ak.'
	dgtaus = np.zeros((int(k_limit))); # Digamma functions and subtraction of Taus
	xxt_qs = np.zeros((int(q))); # X.q'X.q
	uz_sum2s = np.zeros((int(q))); 
	trace_varnus = 0; #Trace of the covariance of Nu (sum of the variances since its diagonal)

	all_k = np.arange(0,k_limit);
	all_q = np.arange(0,q);
	
	for q1 in range(q):
		xxt_qs[q1] = np.sum(np.dot(x[:,q1],x[:,q1]));
		
	previ = datetime.datetime.now();
	# Fit the Model
	print "Fitting the Model...";
	for i in range(iterations):
		print i
		sys.stdout.flush();
		# Update A
		if (learn_a):
			phi_mean,phi_var = update_all_ak_va(kappa1, kappa2, var_n, x, y, phi_mean, nu, k_limit, n, p, q, xxt_qs, threshold, var_a, learn_avar)
			for k1 in range(k_limit):
				exp_phiphis[k1] = np.sum((phi_var[k1,:]))+np.dot(phi_mean[k1,:],phi_mean[k1,:].T);
		phi_mean_traces[:,:,i] = phi_mean;
		phi_var_traces[:,:,i] = phi_var;
		trace_varnus = 0;
		for k1 in range(k_limit):
		    tau_k1[k1] = np.array(alpha/k_limit)+ np.sum(nu[:,k1]);
		    tau_k2[k1] = n + np.sum(1-nu[:,k1]);
		    dgtaus[k1] = scipy.special.digamma(tau_k1[k1]) - scipy.special.digamma(tau_k2[k1])
		    trace_varnus = trace_varnus + (tau_k1[k1]*tau_k2[k1])/(math.pow(tau_k1[k1]+tau_k2[k1],2)*(tau_k1[k1] + tau_k2[k1] + 1))
		tau_traces1[:,i] = tau_k1;
		tau_traces2[:,i] = tau_k2;
		# Update Z
		if (learn_z):
			nu = update_all_zqk(var_n, x, y, nu, phi_mean, phi_var, n, p, q, k_limit, xxt_qs, dgtaus, exp_phiphis, threshold)
		nu_traces[:,:,i] = nu;
		#Update Sigma_A
		if (learn_avar):
			kappa1, kappa2 = update_sigma_a(phi_mean,phi_var,c,d,k_limit,p)
		kappas[:,:,i] = kappa2;
		#Every 100 Iterations, Check Convergence of All Parameters. If Everything has converged we can stop
		if (i % 100 == 0 and i > burn):
		    curi = datetime.datetime.now();
		    print "Iteration "+str(i)+". Time for last iteration: "+delta_time_string(previ,curi);
		    previ = curi;
		    sys.stdout.flush();
		    con_tau1 = geweke2(tau_traces1[:,i-100:i],0.05);
		    con_tau2 = geweke2(tau_traces2[:,i-100:i],0.05);
		    nu_con = geweke3(nu_traces[:,:,i-100:i],0.05);
		    phim_con = geweke3(phi_mean_traces[:,:,i-100:i],0.05);
		    phiv_con = geweke3(phi_var_traces[:,:,i-100:i],0.05);
		    avar_con = geweke3(kappas[:,:,i-100:i],0.05);
		    if (con_tau1):
		        print "Tau1 Converged at iteration "+ str(i)+"!";
		    if (con_tau2):
		        print "Tau2 Converged at iteration "+ str(i)+"!";
		    if (nu_con):
		        print "Nu Converged at iteration "+ str(i)+"!";
		    if (phim_con):
		        print "Phi Mean Converged at iteration "+ str(i)+"!";
		    if (phiv_con):
		        print "Phi Var Converged at iteration "+ str(i)+"!";
		    if (avar_con and learn_avar):
		        print "Prior A Var Converged at iteration "+ str(i)+"!";
		    if (con_tau1 and con_tau2 and nu_con and phim_con and phiv_con and ((avar_con and learn_avar) or not learn_avar)):
		        print "All Converged at iteration "+ str(i)+"!";
		        nu_traces = nu_traces[:,:,0:i];
		        tau_traces1 = tau_traces1[:,0:i];
		        tau_traces2 = tau_traces2[:,0:i];
		        phi_mean_traces = phi_mean_traces[:,:,0:i];
		        phi_var_traces = phi_var_traces[:,0:i];
		        kappas = kappas[:,0:i];
		        iterations = i;
		        break;
		    np.save(output_dir+"nu.npy", nu_traces[:,:,0:i])
		    np.save(output_dir+"phi_mean.npy", phi_mean_traces[:,:,0:i])
		    np.save(output_dir+"phi_var.npy", phi_var_traces[:,0:i])
		    np.save(output_dir+"tau1.npy", tau_traces1[:,0:i])
		    np.save(output_dir+"tau2.npy", tau_traces2[:,0:i])
		    np.save(output_dir+"kappa.npy", kappas)
	return nu_traces,tau_traces1,tau_traces2,phi_mean_traces,phi_var_traces,kappas,iterations
	
######################## MAIN ##############################

np.seterr(over='ignore');

# Input Parameters and Options
explanatory_file = "";
response_file = "";
x = [];
y = [];
k_limit = 100;
iterations = 50;
alpha = 1;
var_a = 0.01;
var_n = 3;
c = 10;
d = 1e-3;#0.25;
a_init = "";
z_init = "";
learn_a = 1;
learn_z = 1;
learn_vara = 1;
output_dir = "./";
maflimit = 0.05;
chrom = "";
info_file = "";
explan_subset = 1;
explan_nr = 0;
makeplots = 0;
threshold = 0.001;
setparams_xval = 0;
provd = 0;
burnin = 100;
ind_list = "";
rand_y = 0;

try:
	options, args = getopt.getopt(sys.argv[1:], "x:y:c:", ["alpha=","iter=","klimit=","ainit=","zinit=","vara=","varn=", "sac=", "sad=", "sa", "sz", "sva", "outdir=","maf=", "info=", "subset=", "list=", "thres=", "nr", "plot", "xval", "permute"]);
	
except getopt.GetoptError, err:
	# print help information and exit:
	print str(err);
	print_help();

for opt, arg in options:
	if opt in ('-x'):
		explanatory_file = arg;
	elif opt in ('-y'):
		response_file = arg;
	elif opt in ('--info'):
		info_file = arg;
	elif opt in ('--klimit'):
		k_limit = int(arg);
	elif opt in ('--iter'):
		iterations = int(arg);
	elif opt in ('--alpha'):
		alpha = float(arg);
	elif opt in ('-c'):
		chrom = arg;
	elif opt in ('--vara'):
		var_a = float(arg);
	elif opt in ('--varn'):
		var_n = float(arg);
	elif opt in ('--sa'):
		learn_a = 0;
	elif opt in ('--sz'):
		learn_z = 0;
	elif opt in ('--sva'):
		learn_vara = 0;
	elif opt in ('--outdir'):
		output_dir = arg;
	elif opt in ('--maf'):
		maflimit = float(arg);
	elif opt in ('--sac'):
		c = float(arg);
	elif opt in ('--sad'):
		d = float(arg);
	elif (opt in '--ainit'):
		a_init = arg;
	elif (opt in '--zinit'):
		z_init = arg;
	elif (opt in '--subset'):
		explan_subset = float(arg);
	elif (opt in '--list'):
		ind_list = arg;
	elif (opt in '--nr'):
		explan_nr = 1;
	elif (opt in '--plot'):
		makeplots = 1;
	elif (opt in '--thres'):
		threshold = float(arg);
	elif (opt in '--xval'):
		setparams_xval = 1;
	elif (opt in '--permute'):
		rand_y = 1;

sampleidsgeno = [];
sampleidsxpr = [];

xbrrr_params = np.array([[1e-3, 0.1, 0.1],[1e-3, 0.1, 0.01],[1e-3, 0.1, 0.001],[1e-3, 0.01, 0.1],[1e-3, 0.01, 0.01],[1e-3, 0.01, 0.001],[1e-3, 0.001, 0.1],[1e-3, 0.001, 0.01],[1e-3, 0.001, 0.001],\
	[1e-2, 0.1, 0.1],[1e-2, 0.1, 0.01],[1e-2, 0.1, 0.001],[1e-2, 0.01, 0.1],[1e-2, 0.01, 0.01],[1e-2, 0.01, 0.001],[1e-2, 0.001, 0.1],[1e-2, 0.001, 0.01],[1e-2, 0.001, 0.001],\
	[1e-1, 0.1, 0.1],[1e-1, 0.1, 0.01],[1e-1, 0.1, 0.001],[1e-1, 0.01, 0.1],[1e-1, 0.01, 0.01],[1e-1, 0.01, 0.001],[1e-1, 0.001, 0.1],[1e-1, 0.001, 0.01],[1e-1, 0.001, 0.001]]); 
xbrrr_l1_params = np.array([[1e-3, 0.1],[1e-3, 0.01],[1e-3, 0.001],[1e-2, 0.1],[1e-2, 0.01],[1e-2, 0.001],[1e-1, 0.1],[1e-1, 0.01],[1e-1, 0.001]]);
xval_parts = 5;
	
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
	
# Read in the Explanatory Variables
if (explanatory_file.endswith('.gen') or explanatory_file.endswith('.gen.gz') or explanatory_file.endswith('.gens.gz') or explanatory_file.endswith('.gens')):
	x,maf,al1,al2,snpid,rsid,snploc,sampleidsgeno = readSNPTESTGenotypes(explanatory_file,maflimit,output_dir);
elif (explanatory_file.endswith('.npy')):
	x = np.load(explanatory_file);
	maf = np.load(str.replace(explanatory_file,'x.npy','maf.npy'));
	al1 = np.load(str.replace(explanatory_file,'x.npy','al1.npy'));
	al2 = np.load(str.replace(explanatory_file,'x.npy','al2.npy'));
	snpid = np.load(str.replace(explanatory_file,'x.npy','snpid.npy'));
	rsid = np.load(str.replace(explanatory_file,'x.npy','rsid.npy'));
	snploc = np.load(str.replace(explanatory_file,'x.npy','snploc.npy'));
else:
	f = open(explanatory_file, 'r');
	reader = csv.reader(f);
	xheader = reader.next();
	xnames = [];
	for row in reader:
		x.append(row[1:]);
		xnames.append([row[0]]);
	f.close();
x = np.array(x,dtype=float);

# Get the Non-Redundant Explanatory Variables
if (explan_nr):
	a = np.ascontiguousarray(x.T)
	unique_a,indices = np.unique(a.view([('', a.dtype)]*a.shape[1]),return_index=True)
	select = np.sort(indices)
	print "Found "+str(len(select))+" Non-Redundant Variables...";
	x = x[:,select];
	maf = maf[select];
	al1 = al1[select];
	al2 = al2[select];
	snpid = snpid[select];
	rsid = rsid[select];
	snploc = snploc[select];
elif (ind_list != ""):
	indices = np.load(ind_list);
	select = np.sort(indices);
	x = x[:,select];
	maf = maf[select];
	al1 = al1[select];
	al2 = al2[select];
	snpid = snpid[select];
	rsid = rsid[select];
	snploc = snploc[select];
select_thismany = int(math.ceil(x.shape[1]*explan_subset));

#Select a Subset of Explanatory Variables
if (explan_subset==1):
	select = np.arange(x.shape[1]);
else:
	select = np.random.choice(x.shape[1],select_thismany,0);	

print "Selected "+str(len(select))+" Explanatory Variables...";
x = x[:,select];
np.save(output_dir+"selectedx.npy", select)
try:
  np.save(output_dir+"maf.npy", maf[select])
  np.save(output_dir+"al1.npy", al1[select])
  np.save(output_dir+"al2.npy", al2[select])
  np.save(output_dir+"snpid.npy", snpid[select])
  np.save(output_dir+"rsid.npy", rsid[select])
  np.save(output_dir+"snploc.npy", snploc[select])
except NameError:
	print "No Gene Info Files To Save";

# Read in the Response Variables
if (response_file.endswith('gex_pccorrected.snptest')):
	genes,sampleidsxpr,y,genesinfo,trans,chroms,loc1,loc2 = readSNPTESTGeneXpr(response_file,info_file,chrom)
elif (response_file.endswith('.npy')):
    y = np.load(response_file);
    genes = np.load(str.replace(response_file,'y.npy','genes.npy'));
    # sampleidsxpr = np.loadtxt(re.sub(response_file,'y.npy','genesinfo.npy'));
    genesinfo = np.load(str.replace(response_file,'y.npy','genesinfo.npy'));
    trans = np.load(str.replace(response_file,'y.npy','trans.npy'));
    # chroms = np.load(re.sub(response_file,'y.npy','chroms.npy'));
    loc1 = np.load(str.replace(response_file,'y.npy','loc1.npy'));
    loc2 = np.load(str.replace(response_file,'y.npy','loc2.npy'));
else:
	f = open(response_file, 'r');
	reader = csv.reader(f);
	yheader = reader.next();
	ynames = [];
	for row in reader:
		y.append(row[1:]);
		ynames.append([row[0]]);
	f.close();
y = np.array(y,dtype=float);

if (len(sampleidsgeno)>0 and len(sampleidsxpr)>0):
	shared,i1,i2 = intersect(sampleidsgeno,sampleidsxpr);
	x = x[i1,:];
	y = y[i2,:];
	
if (rand_y==1):
	np.random.shuffle(y);
	
np.save(output_dir+"x.npy", x)
np.save(output_dir+"y.npy", y)
try:
	np.save(output_dir+"genes.npy", genes)
	np.save(output_dir+"genesinfo.npy", genesinfo)
	np.save(output_dir+"trans.npy", trans)
	np.save(output_dir+"loc1.npy", loc1)
	np.save(output_dir+"loc2.npy", loc2)
except NameError:
	print "No Gene Info Files To Save";

n = y.shape[0];
p = y.shape[1];
q = x.shape[1];

# If an Initial A is Provided, Read it in. Otherwise Initialize Randomly
if (a_init != ""):
	phi_mean = [];
	f = open(a_init, 'r');
	reader = csv.reader(f);
	header = 0;
	for row in reader:
		phi_mean.append(row);
	f.close();
	phi_mean = np.array(phi_mean,dtype=float);
else:
	phi_mean = np.random.normal(0,0.5,(int(k_limit),int(p)));
	
# If an Initial Z is Provided, Read it in. Otherwise Initialize Randomly
if (z_init != ""):
	nu = [];
	f = open(z_init, 'r');
	reader = csv.reader(f);
	header = 0;
	for row in reader:
		nu.append(row);
	f.close();
	nu = np.array(nu,dtype=float);
	nu = nu[select,:];
else:
	nu = np.random.uniform(0, 0.1, (int(q),int(k_limit)));

#Make plots of the data
if (makeplots ==1):
	plt.clf();
	plt.imshow(x,interpolation='none')
	plt.ylabel('Sample')
	plt.xlabel('SNP')
	plt.title('Correct X')
	plt.savefig(output_dir+'CorrectX.png')
	
	plt.clf();
	plt.imshow(y,interpolation='none')
	plt.ylabel('Sample')
	plt.xlabel('Gene')
	plt.title('Correct Y')
	plt.savefig(output_dir+'CorrectY.png')

	plt.clf();
	plt.imshow(nu.T,interpolation='none')
	plt.ylabel('Factor')
	plt.xlabel('SNP')
	plt.title('Correct Z')
	plt.savefig(output_dir+'CorrectZ.png')

	plt.clf();
	plt.imshow(phi_mean.T,interpolation='none')
	plt.ylabel('Gene')
	plt.xlabel('Factor')
	plt.title('Correct A')
	plt.savefig(output_dir+'CorrectA.png')

# If we're setting the parameters via cross-validation, split the data, 
# find the optimal parameters, and fit the model using those optimal parameters
if (setparams_xval==1):
	splitx,splity,full_true_y = splitXYcrossval(x,y,xval_parts)
	
	# Run the Model with the L1 Penalty on A
	if (learn_vara==1):
		rsss = np.zeros(np.shape(xbrrr_l1_params)[0]);
		for xp in range(np.shape(xbrrr_l1_params)[0]):
			full_est_y = np.zeros(np.shape(full_true_y));
			start = 0;
			print "Parameters:"
			print xbrrr_l1_params[xp,:]
			# Fit the model for each each parameter setting for xval partition
			for xvp in range(xval_parts):
				nu_traces,tau_traces1,tau_traces2,phi_mean_traces,phi_var_traces,kappas,numiter = fitModel(iterations,burnin,splitx[xvp],splity[xvp],p,q,k_limit,c,d,phi_mean,nu,xbrrr_l1_params[xp,0],var_a,xbrrr_l1_params[xp,1],learn_a,learn_z,learn_vara, threshold, output_dir);
				ey = (np.dot(splitx[xvp],np.dot(nu_traces[:,:,numiter-1],phi_mean_traces[:,:,numiter-1])));
				full_est_y[start:start+np.shape(ey)[0],:] = ey;
				start = start + np.shape(ey)[0];
			# Compute the RSS for this paramter setting
			rsss[xp] = computeRSS(full_true_y,full_est_y);
			print rsss[xp]
		# Find the combination of parameter settings with the minimum RSS
		print rsss
		min_rss, best_ind = min((val, idx) for (idx, val) in enumerate(rsss));
		alpha = xbrrr_l1_params[best_ind,0];
		var_n = xbrrr_l1_params[best_ind,1];
	# Run the Model without the L1 Penalty on A
	else:
		rsss = np.zeros(np.shape(xbrrr_params)[0]);
		for xp in range(np.shape(xbrrr_params)[0]):
			full_est_y = np.zeros(np.shape(full_true_y));
			start = 0;
			print "Parameters:"
			print xbrrr_params[xp,:]
			# Fit the model for each each parameter setting for xval partition
			for xvp in range(xval_parts):
				nu_traces,tau_traces1,tau_traces2,phi_mean_traces,phi_var_traces,kappas,numiter = fitModel(iterations,burnin,splitx[xvp],splity[xvp],p,q,k_limit,c,d,phi_mean,nu,xbrrr_params[xp,0],xbrrr_params[xp,1],xbrrr_params[xp,2],learn_a,learn_z,learn_vara,threshold,output_dir);
				ey = (np.dot(splitx[xvp],np.dot(nu_traces[:,:,numiter-1],phi_mean_traces[:,:,numiter-1])));
				full_est_y[start:start+np.shape(ey)[0],:] = ey;
				start = start + np.shape(ey)[0];
			# Compute the RSS for this paramter setting
			rsss[xp] = computeRSS(full_true_y,full_est_y);
			print rsss[xp]
		# Find the combination of parameter settings with the minimum RSS
		print rsss
		min_rss, best_ind = min((val, idx) for (idx, val) in enumerate(rsss));	
		alpha = xbrrr_params[best_ind,0];
		var_a = xbrrr_params[best_ind,1];
		var_n = xbrrr_params[best_ind,2];

nu_traces,tau_traces1,tau_traces2,phi_mean_traces,phi_var_traces,kappas,numiter = fitModel(iterations,burnin,x,y,p,q,k_limit,c,d,phi_mean,nu,alpha,var_a,var_n,learn_a,learn_z,learn_vara,threshold,output_dir);
sampled_z = np.array(nu_traces[:,:,numiter-1] > 0.95,dtype=float)
ey = (np.dot(x,np.dot(sampled_z,phi_mean_traces[:,:,numiter-1])));
# ey = (np.dot(x,np.dot(nu_traces[:,:,numiter-1],phi_mean_traces[:,:,numiter-1])));
rss = computeRSS(y,ey)
cv = computeCV(y,ey)
			
if (learn_vara==1):
	var_a = kappas[:,:,numiter-1]/(c+(1/2)-1);

# Write Out the Results
f = open(output_dir+"alpha.csv",'w')
f.write(str(alpha)) 
f.close()
np.save(output_dir+"var_a.npy",var_a)
f = open(output_dir+"var_n.csv",'w')
f.write(str(var_n)) 
f.close()
np.save(output_dir+"nu.npy", nu_traces)
np.save(output_dir+"phi_mean.npy", phi_mean_traces)
np.save(output_dir+"phi_var.npy", phi_var_traces)
np.save(output_dir+"tau1.npy", tau_traces1)
np.save(output_dir+"tau2.npy", tau_traces2)
np.save(output_dir+"kappa.npy", kappas)
sampled_z = np.array(nu_traces[:,:,numiter-1] > 0.95,dtype=float)
ey = (np.dot(x,np.dot(sampled_z,phi_mean_traces[:,:,numiter-1])));
np.savetxt(output_dir+"est_y.csv", ey, delimiter=",")

rank = np.sum(((np.sum(sampled_z > threshold,axis=0)>0) * (np.sum(phi_mean_traces[:,:,numiter-1]> threshold,axis=1)>0)));
f = open(output_dir+"rank.csv",'w')
f.write(str(rank)) 
f.close()
f = open(output_dir+"metrics.csv", 'w');
f.write(str(rss)+","+str(cv)+","+str(rank));
f.close();

# Plot and Save the Convergence and Results
if (makeplots==1):
	fontP = FontProperties();
	fontP.set_name('Arial')
	
	plt.clf();
	plt.imshow(nu_traces[:,:,numiter-1].T,interpolation='none')
	plt.colorbar()
	plt.ylabel('Factor')
	plt.xlabel('SNP')
	plt.title('Estimated Z')
	plt.savefig(output_dir+'EstimatedZ.png')
	
	plt.clf();
	plt.imshow(phi_mean_traces[:,:,numiter-1].T,interpolation='none')
	plt.colorbar()
	plt.ylabel('Gene')
	plt.xlabel('Factor')
	plt.title('Estimated A')
	plt.savefig(output_dir+'EstimatedA.png')
	
	plt.clf();
	plt.imshow((np.dot(x,np.dot(nu_traces[:,:,numiter-1],phi_mean_traces[:,:,numiter-1]))),interpolation='none')
	plt.ylabel('Sample')
	plt.xlabel('Gene')
	plt.title('Estimated Y')
	plt.savefig(output_dir+'EstimatedY.png')
	
	plt.clf()
	plt.plot(np.tile(range(numiter),(k_limit,1)).T,tau_traces1.T);
	plt.plot(np.tile(range(numiter),(k_limit,1)).T,tau_traces2.T);
	plt.ylabel('Tau k')
	plt.xlabel('Iteration')
	plt.title('Traces of Tau')
	plt.savefig(output_dir+'TracesPi.png')
	
	plt.clf()
	for ki in range(k_limit):
		plt.plot(range(numiter),phi_mean_traces[ki,:,:].T);
	plt.ylabel('Phi kp')
	plt.xlabel('Iteration')
	plt.title('Traces of Phi')
	plt.savefig(output_dir+'TracesA.png')
	
	plt.clf()
	for ki in range(k_limit):
		plt.plot(np.tile(range(numiter),(q,1)).T,nu_traces[:,ki,:].T);
	plt.ylabel('Nu qk')
	plt.xlabel('Iteration')
	plt.title('Traces of Nu')
	plt.savefig(output_dir+'TracesZ.png')

