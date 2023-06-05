import numpy as np
import os 
import csv
import sys

# Open velocity (read) and phi (write) files
# Find where vel is null and set corresponding phi line to 1000
# Output file is called map_phi 

nstart = 1000
nend = 30000
nint = 1000

args = sys.argv[1:]
nstart = int(args[0])
nend = int(args[1])
nint = int(args[2])

filelist = []
filelist_vel = []
filelist_phi = []

for i in range(nstart,nend+nint,nint):
  os.system('ls -t1 vel-%08.0d.vtk >> filelist_vel' % i)
  filelist_vel.append('vel-%08.0d.vtk' % i)
  os.system('ls -t1 phi-%08.0d.vtk >> filelist_phi' % i)
  filelist_phi.append('phi-%08.0d.vtk' % i)


for i in range(len(filelist_vel)):

  with open(filelist_vel[i], 'r') as velfile, open(filelist_phi[i], 'r') as phifile:
    vel_lines = velfile.readlines()
    phi_lines = phifile.readlines()

    HEADERS = phi_lines[:10]
    DIMENSIONS = HEADERS[4]
    NX, NY, NZ = int(DIMENSIONS.split()[1]), int(DIMENSIONS.split()[2]), int(DIMENSIONS.split()[3]) 

    PHIMAP = []
    for heads in HEADERS:
      PHIMAP.append(heads.strip())


    vel_lines = np.array(vel_lines[9:])
    phi_lines = np.array(phi_lines[10:])
    n = 0

    for index in range(NX*NY*NZ):
      kl = index // NX*NY
      jl = (index - kl*NX*NY) // NX
      il = index - jl*NX - kl*NX*NY
      vels = vel_lines[index].split()
      phis = phi_lines[index].split()
         
      if vels[0] == "0.000000e+00" and vels[1] == "0.000000e+00" and vels[2] == "0.000000e+00":
        PHIMAP.append(" 1.000000e+02  1.000000e+02")
        n += 1
      else: 
        PHIMAP.append(" " + str(phis[0]) +"  " + str(phis[1]))

  print(n)


  with open('map_'+filelist_phi[i], 'w') as outputfile:
    for line in PHIMAP:
      outputfile.write(line)
      outputfile.write("\n")

if os.path.exists("filelist_vel"): 
  os.remove("filelist_vel")
if os.path.exists("filelist_phi"): 
  os.remove("filelist_phi")
