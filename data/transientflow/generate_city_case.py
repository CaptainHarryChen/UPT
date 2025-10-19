import os
import argparse
import shutil
import random
import time
import pickle
from fluidfoam import readscalar
from fluidfoam import readmesh
from fluidfoam import readvector
import meshio

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot前设置
import matplotlib.pyplot as plt
import io
from PIL import Image

from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedParameterFile import ParsedBoundaryDict

from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Execution.ParallelExecution import LAMMachine

from CityMeshGenerator import generate_mesh,sort_points


def readU(arg):
    i,dest = arg
    return torch.tensor(readvector(dest,str(i),'U'))

def readp(arg):
    i,dest = arg
    return torch.tensor(readscalar(dest,str(i),'p'))

def readPhi(arg):
    i,dest = arg
    return torch.tensor(readscalar(dest,str(i),'phi'))

def prepareCase(src, dest, n_points, velocity, n_cores):
    if os.path.exists(dest) and os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src,dest)
    
    while True:
        failed = False
        try:
            generate_mesh(dest+"mesh.msh",n_points)
            runner = BasicRunner(argv=["gmshToFoam","-case",dest,dest+"mesh.msh"],logname="logifle",noLog=True)
            runner.start()

            f = ParsedBoundaryDict(dest+"constant/polyMesh/boundary")
            f['sides']['type'] = 'symmetry'
            f.writeFile()
            f = ParsedParameterFile(dest+"0/U")
            f['internalField'] = 'uniform ('+str(velocity)+' 0 0)'
            f.writeFile()
            f = ParsedParameterFile(dest+"system/decomposeParDict")
            f['numberOfSubdomains'] = n_cores
            f.writeFile()

            runner = BasicRunner(argv=["decomposePar","-case",dest],logname="logifle",noLog=True)
            runner.start()

        except Exception as e:
            print("retry mesh generation")
            failed = True

        if not failed:
            print("Mesh generated")
            break
        else:
            print("retry mesh generation")
            time.sleep(3)


def get_current_case(parent_directory):
    # Get a list of all directories in the parent directory
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    #check if empty
    if not any(directories):
        return 0
    # Extract numerical parts from directory names and convert to integers
    existing_numbers = [int(d.split('_')[1]) for d in directories if d.startswith("case_") and d[5:].isdigit()]
    # Find the lowest missing directory number
    lowest_missing_number = None
    for i in range(1, max(existing_numbers) + 2):
        if i not in existing_numbers:
            lowest_missing_number = i
            break
    return lowest_missing_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_objects', type=int, help='maximum number of circles/partial circles the case should have (min is 1)')
    parser.add_argument('n_cases', type=int, help='number of cases to be run')
    parser.add_argument('n_cores',type = int, help='number of CPU-cores to use for computation')
    parser.add_argument('empty_case',type=str, help='the empty openfoam case directory')
    parser.add_argument('dest', type=str, help='target directory for the OpenFOAM Cases')
    parser.add_argument('working_dir',type=str, help='working directory for OpenFoam Simulation')
    
    args = parser.parse_args()
    num_points = args.n_objects
    assert num_points>0, "n_objects < 1"
    n_cores = args.n_cores
    assert n_cores>1, "at least two core should be used"
    mpiInformation = LAMMachine(nr=n_cores)
    n_cases = args.n_cases
    src = args.empty_case
    dest = args.dest
    work_dir = args.working_dir
    dest = os.path.join(dest, '')
    work_dir = os.path.join(work_dir, '')
    current_case_number = get_current_case(dest)
    
    init_vel_min, init_vel_max = 2.0, 7.0
    delta_t = [0.05,0.025,0.01,0.005,0.0025,0.001,0.0005,0.00025,0.0001,0.00005,0.000025,0.00001]
    delta_t_index = 0

    print("Working directory is: " + work_dir)
    print("Cases are written to: " + dest)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    while current_case_number < n_cases:
        print("current case: ",str(current_case_number))
        n_points = random.randint(max(1, num_points // 2), num_points)
        velocity = random.uniform(init_vel_min, init_vel_max)
        nr_time_steps = 0
        crash_counter = 0
        prepareCase(src,work_dir,n_points,velocity,n_cores)
        # exit()

        msh = meshio.read(work_dir+"/mesh.msh")
        triangles = msh.cells_dict['triangle'][(msh.points[msh.cells_dict['triangle']][:,:,-1] == 0)[:,0]]
        mesh_points = msh.points
        time.sleep(5)
        
        delta_t_index = 0
        f = ParsedParameterFile(work_dir+"system/controlDict")
        max_time_steps = f['endTime']
        f['deltaT'] = delta_t[delta_t_index]
        f.writeFile()

        try:
            while nr_time_steps < max_time_steps:
                delta_t_index += 1
                if crash_counter > 0:
                    f = ParsedParameterFile(work_dir+"system/controlDict")
                    f['deltaT'] = delta_t[delta_t_index]
                    f.writeFile()

                runner = BasicRunner(argv=["pisoFoam","-case",work_dir],logname="logifle",noLog=True,lam=mpiInformation)
                run_information = runner.start()
                nr_time_steps = run_information['time']
                crash_counter += 1

        except IndexError:
            print("List out of bound, restarting outer loop")
            continue
        
        runner = BasicRunner(argv=["redistributePar","-reconstruct","-case",work_dir],logname="logifle",noLog=True,lam=mpiInformation)
        runner.start()
        current_case_number = get_current_case(dest)
        solution_dir = dest+"/case_"+str(current_case_number)+"/"
        
        for tries in range(10):
            try:
                os.mkdir(solution_dir)
            except OSError as error:
                print("case already claimed by other process, retry")
                current_case_number = get_current_case(dest)
                solution_dir = dest+"/case_"+str(current_case_number)+"/"
            else:
                print("case available, claiming...")
                break
        os.remove(work_dir+"PyFoamState.CurrentTime")
        os.remove(work_dir+"PyFoamState.LastOutputSeen")
        os.remove(work_dir+"PyFoamState.StartedAt")
        os.remove(work_dir+"PyFoamState.TheState")
        
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if name.startswith("processor") and os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    print("Failed to remove {}: {}".format(path, e))
        try:
            shutil.copytree(work_dir, solution_dir, dirs_exist_ok=True)
        except Exception as e:
            print("Failed to copy {} -> {}: {}".format(work_dir, solution_dir, e))
        
        print("solution directory: ", solution_dir)
        current_case_number = get_current_case(dest)
        time.sleep(5)

if __name__=="__main__":
    main()
