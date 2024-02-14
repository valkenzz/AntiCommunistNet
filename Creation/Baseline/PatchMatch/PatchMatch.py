# -*- coding: utf-8 -*-
"""
@author: Valentin meo (for any question or bug : valentin.meo.1@ulaval.ca)
This was adapted from https://github.com/AlexandreSev/Patch-Match autorisation has been given for non comercial use
"""


#import 
import numpy as np
from os import listdir
import numpy as numpy
from scipy.linalg import norm
from skimage import measure
import networkx as nx
import time
import cv2
import cv2 as cv

#seed
np.random.seed(430)

#Size of inpainting
tailleTrou=int(64)

#If True, the algorithm will also try rotated patch
rotation = False

if rotation:
    r_max = 4
else:
    r_max = 1

#Size of Image
imSize=256

#Define the minimum and maximum size of applied patch
psi_min = 22
psi_max = 28

#Define the size of the neighborhood in which the algorithm will search
search_area_size = 64

#Where is located original image
mypath='../tempPatchMatch'

#Where save the fake image
SaveDire = str('/media/seagate/vmeo/detectionDataSet/Baseline/campagne/F/img/')


#All needed functions

#find priority term
def confidence_coefficient(confidence, pt, psi=9):
    result = 0
    for i in range(psi):
        for j in range(psi):
            result += confidence[np.int(pt[0] + i - (psi-1)/2+0.5), np.int(pt[1] + j - (psi-1)/2 +0.5)]
    return float(result)/psi**2

#Normal vector to the contour
def gen_normal_vector(omega):
    tangeantes = [(omega[i+1][0] - omega[i-1][0],omega[i+1][1] - omega[i-1][1]) for i in range(1, len(omega)-1)]
    normals = [(-j/np.sqrt(i*i+j*j), i/np.sqrt(i*i+j*j)) for i,j in tangeantes]
    return normals 

#Creation of the data coefficient
def gen_data_vector(normal, isophote, alpha=255.):
    result = [np.abs(normal[i][0] * isophote[i][0] + normal[i][1] * isophote[i][1])/alpha for i in range(len(isophote))]
    return result

#Create a 3d norm
def norm3d(img, param):
    temp = img.reshape((img.shape[0], img.shape[1], -1))
    shape = temp.shape[2]
    result = 0
    for i in range(shape):
        result += norm(img[:, :, i], param)
    return result

#Create a distance which takes into account the distance to a hole
def distance(img1, img2, is_compared, gauss, param = 1):
    temp = img1.reshape((img1.shape[0], img1.shape[1], -1))
    shape = temp.shape[2]
    result = 0
    for i in range(shape):
        result += norm((img1[:, :, i] - img2[:, :, i]) * gauss, param)
    return result

# Compute the distance for a given center
def create_kernel_local(xc, yc):
    result =np.zeros((psi,psi))
    for i in range(psi):
        for j in range(psi):
            result[i,j] = float((i-xc)**2 + (j-yc)**2) / 2
    m = np.max(result)        
    result = 1 - result.astype(np.float) / m
    return result

#Compute distance for a whole hole
def create_kernel_patch(is_compared):
    result = np.zeros(is_compared.shape)
    counter = 0
    for i in range(is_compared.shape[0]):
        for j in range(is_compared.shape[1]):
            if is_compared[i,j] == 0:
                result += create_kernel_local(i, j)
                counter += 1
    result = result.astype(np.float) / counter
    return result

#Find best match with one core
def findbestmatch(patch, img, is_compared, patch_x, patch_y, psi=9):
    result = 9999
    gauss = create_kernel_patch(is_compared[:, :, 0])
    counter = 0
    for i in range(max(0, int(patch_x - search_area_size)), int(patch_x + search_area_size + 1)):
            for j in range(max(0, int(patch_y - search_area_size)), int(patch_y + search_area_size + 1)):
                for r in range(r_max):
                    counter +=1
                    if counter % 50000 == 0:
                        print("Counter %s"%counter)
                    patch_comp = patch * is_compared
                    if (i<psi/2) | (j<psi/2) | (i>img.shape[0]-psi/2-1) | (j>img.shape[1]-psi/2-1):
                        pass
                    else:
                        img_comp = img[i - psi/2: i + psi/2+1, j - psi/2: j + psi/2+1].copy()
                        img_comp = np.rot90(img_comp, r)
                        
                        if np.all(img_comp[:, :, 0] != -1):
                            img_comp *= is_compared
                            if distance(img_comp, patch_comp, is_compared[:, :, 0], gauss) < result:
                                result = distance(img_comp, patch_comp, is_compared[:, :, 0], gauss) 
                                result_x = i
                                result_y = j
                                result_r = r                                                                
    return result_x, result_y, result_r

#Find the right contour in the picture
def find_good_contour(contours, img):
    for contour in contours:
        pt = contour[len(contour)/2]
        pt = [int(i) for i in pt]
        if np.any(img[pt[0]-1:pt[0]+2, pt[1]-1:pt[1]+2]==[-1,-1,-1]) :
            contour = np.concatenate(([contour[0]], contour, [contour[-1]]))
            contour = contour+0.5
            return True, contour
    return False, []

#Find the maximal value in temp
def pooling(temp):
    return numpy.max(numpy.abs(temp))

#Compute the gradient of a patch with a hole in it
def get_isophote(pts, img, psi=9):
    isophote = []
    for pt in pts:
        x = pt[0]
        y = pt[1]
        temp = img[np.int(x - psi/2): np.int(x + psi/2+1), np.int(y - psi/2): np.int(y + psi/2+1)]
        grad = numpy.gradient(numpy.ma.masked_where(temp == -1, temp))
        isophote.append((pooling(grad[0]), pooling(grad[1])))
    return isophote

#Small usefull function
def should_be_connected(n1, n2):
    if (len(n1) == 2) & (len(n2) == 2):
        return( ((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2) == 1 )
    else:
        raise NameError("Wrong node")

#Compute the term to term addition of two tuples
def tuple_add(pt1, pt2):
    return (int(pt1[0] + pt2[0]), int(pt1[1] + pt2[1]))

#Compute the term to term difference of two tuples
def tuple_minus(pt1, pt2):
    return (pt1[0] - pt2[0], pt1[1] - pt2[1])

#Compute the weight of an edge
def edge_weight(pt1_ini, pt2_ini, old_patch, new_patch, demi=False):   
    if demi:
        diff = tuple_minus(pt2_ini, pt1_ini)

        pt1 = (int(pt1_ini[0]), int(pt1_ini[1]))
        pt2 = tuple_add(pt2_ini, diff)
    else:
        pt1 = (int(pt1_ini[0]), int(pt1_ini[1]))
        pt2 = (int(pt2_ini[0]), int(pt2_ini[1]))
    
    return (norm3d(old_patch[pt1].reshape((1, 1, -1))-new_patch[pt1].reshape((1, 1, -1)), 2) + 
        norm3d(old_patch[pt2].reshape((1, 1, -1))-new_patch[pt2].reshape((1, 1, -1)), 2))
        
#Transform a tuple into a tuple of integers
def tuple_in(t, l):
    return (t in l) | ((t[1], t[0]) in l)

#Transform the patch probleme into a graph
def create_graph(img_ini, pt_ini, new_patch, cut_edges, psi=9):
    img = img_ini.reshape((img_ini.shape[0], img_ini.shape[1], -1))
    pt = (int(pt_ini[0] + 0.5), int(pt_ini[1] + 0.5))
    old_patch = img[pt[0] - psi/2 - 1: pt[0] + psi/2 + 2, pt[1] - psi/2 - 1: pt[1] + psi/2 + 2]
    G=nx.Graph()
    G.add_node("old")
    G.add_node("new")
    for i in range(pt[0] - psi/2, pt[0] + psi/2+1):
        firsti = (i == pt[0] - psi/2)
        lasti = (i == pt[0] + psi/2)
        for j in range(pt[1] - psi/2, pt[1] + psi/2+1):
            firstj = (j == pt[1] - psi/2)
            lastj = (j == pt[1] + psi/2)
            if img[i, j, 0] != -1:
                G.add_node((i, j))
                if not firsti:
                    if ((img[i-1, j, 0] == -1) & ((i-1, j) not in G.nodes())):
                        G.add_node((i-1, j))
                        G.add_edge((i-1, j), 'new', weight=np.inf)
                else:
                    G.add_node((i-1, j))
                    G.add_edge((i-1, j), 'old', weight=np.inf)
                if not lasti:
                    if ((img[i+1, j, 0] == -1) & ((i+1, j) not in G.nodes())):
                        G.add_node((i+1, j))
                        G.add_edge((i+1, j), 'new', weight=np.inf)
                else:
                    G.add_node((i+1, j))
                    G.add_edge((i+1, j), 'old', weight=np.inf)
                if not firstj:
                    if ((img[i, j-1, 0] == -1) & ((i, j-1) not in G.nodes())):
                        G.add_node((i, j-1))
                        G.add_edge((i, j-1), 'new', weight=np.inf)
                else:
                    G.add_node((i, j-1))
                    G.add_edge((i, j-1), 'old', weight=np.inf)
                if not lastj:
                    if ((img[i, j+1, 0] == -1) & ((i, j+1) not in G.nodes())):
                        G.add_node((i, j+1))
                        G.add_edge((i, j+1), 'new', weight=np.inf)
                else:
                    G.add_node((i, j+1))
                    G.add_edge((i, j+1), 'old', weight=np.inf)
                    
                for ibis, jbis in [(i-1, j), (i+1,j), (i, j-1), (i, j+1)]:
                    if (not tuple_in(((i,j), (ibis, jbis)), G.edges())) & ((ibis,jbis) in G.nodes()):
                        if ((i,j), (ibis, jbis)) in cut_edges:
                            idemi, jdemi = (float(i+ibis)/2, float(j+jbis)/2)
                            G.add_node((idemi, jdemi))
                            G.add_edge((idemi,jdemi), 'new', attr_dict=cut_edges)
                            G.add_edge((i, j), (idemi, jdemi), weight=edge_weight((i - pt[0] + psi/2 + 1, j - pt[1] + psi/2 + 1), 
                                                                 (idemi - pt[0] + psi/2 + 1, jdemi - pt[1] + psi/2 + 1),
                                                                 old_patch, new_patch, demi=True))
                            G.add_edge((ibis, jbis), (idemi, jdemi), weight=edge_weight((ibis - pt[0] + psi/2 + 1, jbis - pt[1] + psi/2 + 1), 
                                                                 (idemi - pt[0] + psi/2 + 1, jdemi - pt[1] + psi/2 + 1),
                                                                 old_patch, new_patch, demi=True))
                        else:
                            G.add_edge((ibis, jbis), (i,j), weight=edge_weight((i - pt[0] + psi/2 + 1, j - pt[1] + psi/2 + 1), 
                                                                     (ibis - pt[0] + psi/2 + 1, jbis - pt[1] + psi/2 + 1),
                                                                     old_patch, new_patch))
    return G

#Find edge which have been cut
def find_cut_edges(G, set_old, dico_cut_edges):
    list_just_cut = []
    for edge in G.edges(data=True):
        node1 = edge[0]
        node2 = edge[1]
        if (node1!='old') & (node1!='new') & (node2!='old') & (node2!='new'):
            if ((node1 in set_old) & (node2 not in set_old)) | ((node1 not in set_old) & (node2 in set_old)):
                if np.floor(node1[0]) != node1[0]:
                    node1 = (2 * node1[0] - node2[0], node1[1])
                elif np.floor(node1[1]) != node1[1]:
                    node1 = (node1[0], 2 * node1[1] - node2[1])
                elif np.floor(node2[0]) != node2[0]:
                    node2 = (2 * node2[0] - node1[0], node2[1])
                elif np.floor(node2[1]) != node2[1]:
                    node2 = (node2[0], 2 * node2[1] - node1[1])
                dico_cut_edges[(node1, node2)]= edge[2]["weight"]
                dico_cut_edges[(node2, node1)]= edge[2]["weight"]
                list_just_cut.append((node1, node2))
    return dico_cut_edges, list_just_cut

#Remove deleted edges
def update_cut_edges(cut_edges, new_set):
    new_cut_edges = {}
    for edge in cut_edges:
        node1 = edge[0]
        node2 = edge[1]
        if (node1 not in new_set) | (node2 not in new_set):
            new_cut_edges[edge]= cut_edges[edge]
    return new_cut_edges

#Mix two patches without blur
def clean_mix(old_patch, new_patch, set_old, patch_x, patch_y, psi):
    applied_patch = new_patch.copy()
    for pt in set_old:
        if (pt!='old'):
            if (np.int(pt[0] - patch_x + psi/2)>=0) & (np.int(pt[0] - patch_x + psi/2)<psi) & \
                (np.int(pt[1] - patch_y + psi/2)>=0) & (np.int(pt[1] - patch_y + psi/2)<psi):
                applied_patch[np.int(pt[0] - patch_x + psi/2), np.int(pt[1] - patch_y + psi/2)] = \
                                old_patch[np.int(pt[0] - patch_x + psi/2 ), np.int(pt[1] - patch_y + psi/2 )]
    return applied_patch

#Mix two patches with blur
def blur_mix(img_ini, new_patch, enlarged_new_patch, set_old, just_cut_edges, patch_x, patch_y, psi):
    img = img_ini.reshape((img_ini.shape[0], img_ini.shape[1], -1))
    pt = (int(patch_x + 0.5), int(patch_y + 0.5))
    old_patch = img[pt[0] - psi/2: pt[0] + psi/2 + 1, pt[1] - psi/2: pt[1] + psi/2 + 1]
    enlarged_old_patch = img[pt[0] - psi/2 - 1: pt[0] + psi/2 + 2, pt[1] - psi/2 - 1: pt[1] + psi/2 + 2]
    applied_patch = img[pt[0] - psi/2 - 1: pt[0] + psi/2 + 2, pt[1] - psi/2 - 1: pt[1] + psi/2 + 2]
    applied_patch[1:-1, 1:-1] = clean_mix(old_patch, new_patch, set_old, patch_x, patch_y, psi)
    for edge in just_cut_edges:
        node1 = edge[0]
        node2 = edge[1]
        value11, value12 = np.int(node1[0] - patch_x + psi/2 + 1), np.int(node1[1] - patch_y + psi/2 + 1)
        value21, value22 = np.int(node2[0] - patch_x + psi/2 + 1), np.int(node2[1] - patch_y + psi/2 + 1)
        if node1 in set_old:
            applied_patch[value11, value12] = 0.66 * enlarged_old_patch[value11, value12] + 0.34 * enlarged_new_patch[value11, value12]
            applied_patch[value21, value22] = 0.34 * enlarged_old_patch[value21, value22] + 0.66 * enlarged_new_patch[value21, value22]
        else:
            applied_patch[value11, value12] = 0.34 * enlarged_old_patch[value11, value12] + 0.66 * enlarged_new_patch[value11, value12]
            applied_patch[value21, value22] = 0.66 * enlarged_old_patch[value21, value22] + 0.34 * enlarged_new_patch[value21, value22]
    return applied_patch[1:-1, 1:-1]




def PatchMatch(marged_img,confidence,psi_min,psi_max,search_area_size,r_max):
    #Initialisation
    time_init = time.time()
    contours = measure.find_contours(marged_img[:,:,0], 0.8)
    criteria, contour = find_good_contour(contours, marged_img)
    nb_iter = 0
    cut_edges = {}

    
    while criteria:
        if not nb_iter%1== 0:
            print("Iteration " + str(nb_iter))
        nb_iter += 1
 
        #Choose of psy
        global psi
        psi = numpy.random.randint(psi_min, psi_max) / 2 * 2 + 1
        
        if False:
            print("psi = %s"%psi)

        #Define the contour as tuple
        omega=[(i[0], i[1]) for i in contour]

        #Create confidence coefficient for each point
        confidence_vector = [confidence_coefficient(confidence, pt, psi=psi) for pt in omega][1:-1]
        

        #Compute isophote for each point
        isophote_vector = get_isophote(omega, marged_img)[1:-1]

        #Compute the normal of the border vector for each point
        normal_vector = gen_normal_vector(omega)

        #Compute de data term for each point
        data_vector = gen_data_vector(normal_vector, isophote_vector, alpha=10.)

        #Compute the priority term for each term
        priority_vector = [data_vector[i] * confidence_vector[i] for i in range(len(data_vector))]

        #Find the point with the highest priority
        m = max(priority_vector)
        indices = [ i for i,j in enumerate(priority_vector) if j==m]
        indice = indices[len(indices)/2]
        patch_x, patch_y = omega[indice+1]

        patch_x = int(patch_x + 0.5)
        patch_y = int(patch_y + 0.5)
        

        if False:
            print("Confidence "+str(confidence_vector[indice]) )
            print("Data "+str(data_vector[indice]))
            print("Priority "+str(priority_vector[indice]))

        #create the patch to find
        patch = marged_img[np.int(patch_x - psi/2): np.int(patch_x + psi/2+1), np.int(patch_y - psi/2): np.int(patch_y + psi/2+1)]

        #Compute the area to compare between the previous patch and the area on the picture
        is_compared = marged_img[np.int(patch_x - psi/2): np.int(patch_x + psi/2+1), np.int(patch_y - psi/2): np.int(patch_y + psi/2+1)]!=-1
        
        result_x, result_y, result_r = findbestmatch(patch, marged_img, is_compared, patch_x, patch_y, psi=psi)

        #create the patch to apply
        true_patch = marged_img[result_x - psi/2: result_x + psi/2+1, result_y - psi/2: result_y + psi/2+1].copy()
        true_patch = np.rot90(true_patch, result_r)
        enlarged_true_patch = marged_img[result_x - psi/2-1: result_x + psi/2+2,
                                            result_y - psi/2-1: result_y + psi/2+2].copy()
        enlarged_true_patch = np.rot90(enlarged_true_patch, result_r)
        
        #Create the graph associated
        G = create_graph(marged_img, (patch_x, patch_y), enlarged_true_patch, cut_edges, psi=psi)
        sets = (nx.minimum_cut(G, 'old', 'new', capacity="weight"))

        #Compute the two separated sets 
        if 'old' in sets[1][0]:
            set_old = sets[1][0]
            set_new = sets[1][1]
        else:
            set_old = sets[1][1]
            set_new = sets[1][0]

        #Find cut edges    
        cut_edges, just_cut_edges = find_cut_edges(G, set_old, cut_edges)


        cut_edges = update_cut_edges(cut_edges, set_new)

        #Modify the true patch
        applied_patch = blur_mix(marged_img, true_patch, enlarged_true_patch, set_old, just_cut_edges, patch_x, patch_y, psi)

        #Apply the patch
        marged_img[np.int(patch_x - psi/2 + 0.5): np.int(patch_x + psi/2+1 + 0.5), 
                   np.int(patch_y - psi/2 + 0.5): np.int(patch_y + psi/2+1 + 0.5)] = applied_patch

        #Update the condidence matrix, the contour and the looping criteria
        m = confidence_vector[indice]
        confidence[np.int(patch_x - psi/2): np.int(patch_x + psi/2+1), np.int(patch_y - psi/2): np.int(patch_y + psi/2+1)]=m
        contours = measure.find_contours(marged_img[:, :, 0], 0.8)
        criteria, contour = find_good_contour(contours, marged_img)

    
    time_taken = time.time() - time_init    
    print("Done in %s seconds"%time_taken)
    return marged_img


#Applie this to the images
a=listdir(mypath)
b=[]
for i in a:
    b.append([(str(mypath)+ str("/")+str(i)),i])

conteuur=0
for i in b:
	img=i[0]
    
    #We get the save path
	nom=i[1]
	chemin=str(SaveDire+str(nom))
    
    #read image
	img = cv.imread(img)
	img = cv2.resize(img, (imSize,imSize), interpolation = cv2.INTER_AREA)
	img.reshape((img.shape[0], img.shape[1], -1))
	shape = img.shape[2]

	# Create the black margin around the image with size pf psi_max on each edge
	marged_img = (np.zeros((img.shape[0] + 2 * psi_max, img.shape[1] + 2 * psi_max, shape)) + 2).astype(int)
	marged_img[psi_max: - psi_max, psi_max:-psi_max] = img.astype(int).copy()

	#size of hole : 
	mid=int(marged_img.shape[0]/2)

	#Create a hole
	marged_img[(mid)-tailleTrou:(mid)+tailleTrou, (mid)-tailleTrou:(mid)+tailleTrou] = - np.ones(shape)

	#Create the confidence matrix
	confidence = np.ones((marged_img.shape[0], marged_img.shape[1]))
	confidence[marged_img[:,:,0] == -1]=0
	confidence[(marged_img[:,:,0] == 2) * (marged_img[:,:,1] == 2) * (marged_img[:,:,2] == 2)]=0


	h=PatchMatch(marged_img,confidence,psi_min,psi_max,search_area_size,r_max)
	h=h[psi_max:(h.shape[0]-psi_max),psi_max:(h.shape[0]-psi_max)]
	cv2.imwrite(chemin, h) 

	conteuur=conteuur+1
	print(conteuur)
	    







