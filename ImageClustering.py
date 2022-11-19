import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def colorQuantize(img, k):
    flat_img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    q_img = np.ones((img.shape[0],img.shape[1],img.shape[2]))
    
    #initialize random means
    indicesy = np.random.choice(range(0,img.shape[0]), size=k, replace=False)
    indicesx = np.random.choice(range(0,img.shape[1]), size=k, replace=False)
    means = np.ones((k,3))
    for i in range(k):
        means[i] = img[indicesy[i]][indicesx[i]]

    in_progress = True
    while in_progress:
        #initialize list of lists to hold clusters
        distances = np.ones((img.shape[0]*img.shape[1], k))
        clusters = []
        for i in range(k):
            clusters.append([])

        #generate clusters based on means
        for i in range(k):
            temp = flat_img - means[i]
            temp = np.square(temp)
            temp = np.sum(temp, axis=1)
            distances[:,i] = temp
        cluster_assignment = np.argmin(distances, axis=1)
        for i in range(cluster_assignment.shape[0]):
            clusters[cluster_assignment[i]].append(flat_img[i])
            
            
        #turn each cluster into a np array
        for cluster in clusters:
            cluster = np.asarray(cluster)

        #find new means
        avg_change = 0
        for i in range(k):
            avg_change += np.mean(np.abs(means[i] - np.mean(np.asarray(clusters[i]),axis=0)))
            means[i] = np.mean(np.asarray(clusters[i]), axis=0)
        avg_change = avg_change/k

        #compare to old means
        if avg_change < 2.5:
            in_progress = False

    #calculate final clusters
    distances = np.ones((img.shape[0]*img.shape[1], k))
    for i in range(k):
        temp = flat_img - means[i]
        temp = np.square(temp)
        temp = np.sum(temp, axis=1)
        distances[:,i] = temp
    cluster_assignment = np.argmin(distances, axis=1)
    for i in range(cluster_assignment.shape[0]):
        flat_img[i] = means[cluster_assignment[i]]
    q_img = flat_img.reshape(img.shape[0],img.shape[1],img.shape[2])
        
    q_img = q_img.astype(int)
    return q_img

if len(sys.argv) != 4:
    print("There must be 3 arguments give: the path to the input image, the path for the output image, and the number of clusters to use.")
    sys.exit() 

inputPath = sys.argv[1]
outputPath = sys.argv[2]
clusters = int(sys.argv[3])

img = cv2.imread(inputPath)
q_img = colorQuantize(img, clusters)
cv2.imwrite(outputPath,q_img)




