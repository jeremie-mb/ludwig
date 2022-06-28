import csv
import numpy as np
import matplotlib.pyplot as plt

plot = False

NVESICLES = 1
NATOMS = 241

RADIUS0 = 5.2796821
RADIUS = 8.0

icosphere_small_l = 0.1821
icosphere_large_l = 0.2060
epsilon = 1e-3

# Choose bond length
BOND_LENGTH = 1.0

# or choose vesicle radius
BOND_LENGTH = RADIUS / RADIUS0

print(str(BOND_LENGTH) + "\n" + str(BOND_LENGTH*icosphere_large_l/icosphere_small_l) + "\n" + str(BOND_LENGTH*RADIUS0))

nbonds= 3
nbonds2 = 3
nbonds3 = 3

XSHIFT = 15
YSHIFT = 15
ZSHIFT = 15

#M27 = np.array([0.061374, -0.467955, 0.843134])
M27 = np.array([-0.218611, 0.822090, 0.458258])
M = np.array([1, 0, 0]) # Vesicle oriented towards X (hole towards -X)

M = M / np.sqrt(np.sum(M**2))
M27 = M27 / np.sqrt(np.sum(M27**2))
 
v = np.cross(M, M27)
s = np.sqrt(np.sum(v**2))
c = np.dot(M27, M)

matvx = np.zeros((3,3))
matvx[0][1] = -v[2]
matvx[0][2] = v[1]
matvx[1][0] = v[2]
matvx[1][2] = -v[0]
matvx[2][0] = -v[1]
matvx[2][1] = v[0]
matvx2 = np.matmul(matvx, matvx)
R = np.eye(3) + matvx + matvx2 * (1/(1+c)) 


# Additional attributes 
indices = np.arange(1,NATOMS+1,1,dtype=int)

nConnec = np.zeros((NATOMS*NVESICLES), dtype = int)
nConnec2 = np.zeros((NATOMS*NVESICLES), dtype = int)
nConnec3 = np.zeros((NATOMS*NVESICLES), dtype = int)
Connec = np.zeros((NATOMS, nbonds), dtype = int)
Connec2 = np.zeros((NATOMS, nbonds2), dtype = int)
Connec3 = np.zeros((NATOMS, nbonds3), dtype = int)
iscentre = np.zeros((NATOMS*NVESICLES), dtype = int)
ishole = np.zeros((NATOMS*NVESICLES), dtype = int)
indexcentre = np.zeros((NATOMS*NVESICLES), dtype = int)

def file_to_array(filename):
  x, y, z = [], [], []

  f = open(filename, "r")
  for line in f.readlines()[:]:
    x.append(line.split()[0])
    y.append(line.split()[1])
    z.append(line.split()[2])

    x = list(map(float, x))
    y = list(map(float, y))
    z = list(map(float, z))
  coords = np.concatenate(([x],[y],[z]))
  return(np.array(coords))

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def rescale(arr, factor):
  arr = np.array(arr) 
  shape = np.shape(arr)
  crepe = np.array([factor * arr.flatten()[i] for i in range(len(arr.flatten())) ] )
  return crepe.reshape(shape)
  
def dist(x,y):
  dr = np.sqrt(np.sum((x-y)**2))
  return dr



# Coordinates
xyz = file_to_array("hexasphere.xyz")


# Connectivities
for i in range(NATOMS):
  bondmade = 0
  bondmade2 = 0
  for j in range(NATOMS):
    if i == j: continue

    dr = dist(xyz.T[i], xyz.T[j])

    if dr < icosphere_small_l + epsilon and bondmade < nbonds:
      Connec[i][bondmade] = np.int_(j)
      nConnec[i] += 1
      bondmade += 1

    if icosphere_large_l - epsilon < dr < icosphere_large_l + epsilon and bondmade < nbonds2:
      Connec2[i][bondmade2] = np.int_(j)
      nConnec2[i] += 1
      bondmade2 += 1 

for i in range(len(Connec)):
  for j in range(len(Connec[i])):
    if Connec[i][j] == 0: continue
    Connec[i][j] += 1
for i in range(len(Connec2)):
  for j in range(len(Connec2[i])):
    if Connec2[i][j] == 0: continue
    Connec2[i][j] += 1
for i in range(len(Connec3)):
  for j in range(len(Connec3[i])):
    if Connec3[i][j] == 0: continue
    Connec3[i][j] += 1

Connec = np.array(Connec)
Connec2 = np.array(Connec2)
Connec3 = np.array(Connec3)


# Renormalize distances so that the smallest of the two harmonic bonds has l_0=1
factor = BOND_LENGTH / icosphere_small_l
xyz = rescale(xyz, factor)

#Other attributes
iscentre[0] = 1 #0, NATOMS, etc...
ishole[240] = 1 #0, NATOMS, etc...
indexcentre[0:NATOMS + 1] = 1 #(0, NATOMS-1), (NATOMS, 2*NATOMS), etc..

xyzt = xyz.T
for i, vec in enumerate(xyzt):
  newvec = np.dot(R.T, vec)
  xyz[0][i] = newvec[0]
  xyz[1][i] = newvec[1]
  xyz[2][i] = newvec[2]

xyz[0, :] += XSHIFT
xyz[1, :] += YSHIFT
xyz[2, :] += ZSHIFT

table = np.column_stack((indices, xyz.T, nConnec, Connec, nConnec2, Connec2, nConnec3, Connec3, iscentre.T, ishole.T, indexcentre.T))
np.savetxt("latticeHexasphere.txt", table, fmt = '%3d     %3f %3f %3f      %3d %3d %3d %3d     %3d %3d %3d %3d     %3d %3d %3d %3d     %3d %3d %3d')

if plot:
  dists = []
  coord0 = xyz[:,0]
  for coord in xyz.T[1::]:
    dist = np.sqrt(np.sum((coord0 - coord)**2 ))
    dists.append(dist)

  print(np.mean(dists), np.max(dists), np.min(dists))


