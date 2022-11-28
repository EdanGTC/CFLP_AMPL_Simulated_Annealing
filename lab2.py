import pandas as pd
import numpy as np
import random
from amplpy import AMPL, Environment
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import argparse
import timeit

import os

import warnings
warnings.filterwarnings('ignore')

#function used for open de file path for ampl
def get_path():
  file = open('path.txt','r')
  return file.readlines()[0]

#function used for assign data to a respectives variables from a given instance 
def get_data_from_file(file):
  # Using readLines()
  file1 = open(file, 'r')
  Lines = file1.readlines()

  #get number of clients and centers, then pop line
  first_line = Lines[0]
  n_center = int(first_line.split(" ")[1])
  n_client = int(first_line.split(" ")[2]) 
  Lines.pop(0)

  centers = [] 
  clients = [] #array of clients with his  

  #object with type center which cointains capacity and cost to open
  center = {
    "capacity": 0,
    "cost_to_open": 0,
  }
  client = {  #object with type client which contains demand and array of cost
    "demand": 0,
    "costs": [] 
  }

  #get capacity and cost_to_open of n centers, then pop lines
  for center_line in Lines[:int(n_center)]:
    aux_center = dict(center)
    if center_line.split(" ")[1].replace('.','',1).isdigit():
      aux_center["capacity"] = int(center_line.split(" ")[1])
    else:
      aux_center["capacity"] = parser.parse_args().capacity
    aux_center["cost_to_open"] = int(center_line.split(" ")[2].replace(".",""))
    centers.append(aux_center)
  Lines = Lines[n_center:]

  cost_list=[]
  demands = []
  cont_demand_cost = 0 #iterate between even and odd refering to get demand and list of costs
  cont_client_center = 0 #cont for stop at amount of costs  
  
  #loop to go through the file line by line
  for line in Lines:
    #flag to get demand of client
    if cont_demand_cost % 2 == 0:
      demands.append( int(line.split(" ")[1]) )
      cont_demand_cost += 1
      continue
    #flag to get list of costs
    aux_line = line.split(" ")
    aux_line.pop(0)
    aux_line.pop()

    for i in aux_line:
      cost_list.append(float(i))
    cont_client_center += len(aux_line)

    if cont_client_center >= n_center:
      cont_demand_cost += 1
      cont_client_center = 0
  #distribute cost for every client
  for demand_per_client in demands:
    aux_client = dict(client)
    aux_client["demand"] = demand_per_client
    aux_client["costs"] = cost_list[0:n_center]
    cost_list = cost_list[n_center:]
    clients.append(aux_client)

  # get data from dictionaries
  demands = []
  for client in clients:
      demands.append(client["demand"])

  capacities = []
  open_costs = []

  for center in centers:
      capacities.append(center["capacity"])
      open_costs.append(center["cost_to_open"])

  return n_center,n_client, centers, clients, demands, capacities, open_costs

def parse_param(name_variable, value):
    return f'param {name_variable} := {value} ;\n'

def parse_array(name_array,array):
    #concatenate aux to array to ampl assign
    aux = f'param {name_array} := '
    for i in range(0,len(array)):
        aux += f'{i+1} {array[i]}    '
    aux += f';\n'

    return aux

def parse_matrix(name_array,n_center,clients):
    #concatenate aux to array to ampl assign
    aux = f'param {name_array} : '

    #concatenate index of rows
    for index in range(0,n_center):
        aux += f'{index+1}    '

    aux += f':= \n'

    #concatenate index of columns and costs
    for index in range(0,len(clients)):
        # print("index", index, ": ")
        aux += f'{index+1}    '
        for cost in clients[index]["costs"]:
            aux += f'{cost}    '

        if(index == len(clients)-1):
            continue
        aux += f'\n'

    aux += f';\n'

    return aux

#define a new  array of possible combination of open centers 
def random_solution_centers(n_center):
    open_centers = np.zeros(n_center)
    while np.sum(open_centers) <= 9: #open centers have to be more than 9
        open_centers = np.random.randint(2, size=n_center)
    return open_centers


# parse data to .dat file
def save_data_AMPL(file,n_client, n_center, open_costs, capacities, demands, clients):
  print("FILE")
  print(os.path.dirname(__file__) + f'/dats/{file.split("/")[1].split(".")[0]}.dat')
  #calling the function to parse the data and write the result in the new fie with the syntax of ampl
  with open(os.path.dirname(__file__) + f'/dats/{file.split("/")[1].split(".")[0]}.dat', 'w') as f:
        f.write(parse_param("cli",n_client))
        f.write(parse_param("loc",n_center))
        f.write(parse_array("FC",open_costs))
        f.write(parse_array("ICap",capacities))
        f.write(parse_array("dem",demands))
        f.write(parse_matrix("TC",n_center, clients))
        f.write(parse_array("x",random_solution_centers(n_center)))

#define a new  array of possible combination of open centers 
def random_solution_centers(n_center):
    open_centers = np.zeros(n_center)
    while np.sum(open_centers) <= 1: #open centers have to be more than 1
        open_centers = np.random.randint(2, size=n_center) #the result of random could be 1 or 0 
    return open_centers

#Function used to generate a new combination of 1 and 0 using the iterations as condition to define the probability of change the current array
def new_solution(current_config_centers, iterations, max_iterations):
      new_solution = current_config_centers.copy()
      l=len(current_config_centers)
      aux_indexes=[]
      #amount of index to change decreases as the iterations decreases
      if iterations > max_iterations*0.8:
        aux_indexes= random.sample(range(0, l-1), int(l*0.8))
      elif iterations > max_iterations*0.6:
        aux_indexes= random.sample(range(0, l-1), int(l*0.6))
      elif iterations > max_iterations*0.4:
        aux_indexes= random.sample(range(0, l-1), int(l*0.4))
      elif iterations > max_iterations*0.2:
        aux_indexes= random.sample(range(0, l-1), int(l*0.2))
      else:
        aux_indexes= random.sample(range(0, l-1), 1)
      #change values of solution
      for i in range(0, len(aux_indexes)):
        if new_solution[aux_indexes[i]] == 1:
          new_solution[aux_indexes[i]]=0
        else:
          new_solution[aux_indexes[i]]=1

      return new_solution

#function used to calculate fitness and solve the assign problem with ampl
def objective_function(open_centers,open_costs):

  X_open_centers_AMPL.set_values(open_centers)
  ampl.solve()

  fitness = float(ampl.get_current_objective().value())
  #amount of cost of to open the given array of open centers 
  cost_open_centers = np.sum(open_centers[:]*open_costs[:])

  #result of CFLP heuristic + assing AMPL
  res_CFLP = cost_open_centers + fitness 
  return res_CFLP

#metaheuristic used to find a good combination of open centers using tempeture, alpha, number of centers, array of centers 
#and returns an array of best solutions and total costs
def simulated_annealing(n_center):

  #Const metaheuristic
  temp_start = 40000
  temp_min   = 10
  temp_aux   = temp_start
  alpha      = 0.95
  
  current_config_centers = random_solution_centers(n_center) #Call the function to generate a random solution to start
  best_solution = current_config_centers.copy() 
  new_config_centers=0
  fitness = objective_function(current_config_centers, open_costs) #calculate the sum of fitness and cost
  fitness_minimized = fitness
  prob = 0 #Probability of choose the worst option
  max_iterations = parser.parse_args().iterations #Number of iterations of the metaheuristic
  iterations = max_iterations
  total_costs = []

  #loop used to find the best solution changing the array current_config_centers and fitness
  while ( iterations > 0 ) and (temp_aux > temp_min):
      new_config_centers = new_solution(current_config_centers, iterations, max_iterations) #random_solution_centers(n_center)
      new_fitness= objective_function(new_config_centers,open_costs)
      delta=new_fitness-fitness

      total_costs.append(ampl.get_value('Total_Cost'))

      if delta<0:
        #updates of values of new array of open center and the fitness related to the center
          current_config_centers=new_config_centers
          fitness=new_fitness
      else:
          prob = np.exp(-delta/temp_aux)#avoid get stuck on a local optimum
          randomProb = random.uniform(0, 1)
          if randomProb < prob:
              current_config_centers=new_config_centers[:]
              fitness=new_fitness

      if fitness - fitness_minimized < 0:
          best_solution = current_config_centers[:]
          fitness_minimized = fitness
      
      temp_aux = temp_aux*alpha #update of tempeture
      
      iterations = iterations-1 #update de iterations

  return best_solution[:],total_costs


# handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=500, help="Number of iterations")
parser.add_argument("-c", "--capacity", type=int, default=7250, help="Center capacity")
parser.add_argument("-p", "--plot", default=False, action='store_true', help="Show plot")
parser.add_argument("-r", "--relaxed", default=False, action='store_true', help="Relax y constrain between [0,1]")
parser.add_argument("-f", "--file", type=str, default="cap134.txt", help="Filename")
args = parser.parse_args()

#------------MAIN-----------------
file = f'instances/{parser.parse_args().file}'
n_center,n_client, centers, clients, demands, capacities, open_costs = get_data_from_file(file)

save_data_AMPL(file,n_client, n_center, open_costs, capacities, demands, clients)
fcost=600000
#to get runtime
start = timeit.default_timer()
#create an AMPL instance
ampl = AMPL(Environment(get_path()))
ampl.reset()
    
ampl.set_option("presolve", False)
#interpret the two files
if (parser.parse_args().relaxed):
  ampl.read('models/CFLP_model_relaxed.mod')
else:
  ampl.read('models/CFLP_model.mod')

ampl.read_data(os.path.dirname(__file__) + f'/dats/{file.split("/")[1].split(".")[0]}.dat')

#instance of x parameter to change values of model
X_open_centers_AMPL = ampl.get_parameter('x')

best_solution_centers,total_costs  = simulated_annealing(n_center) #call the function of simulated annealing with n_center as parameter

#values to plot
x_total_costs = np.arange(0, len(total_costs), dtype=int)
best_cost = min(total_costs)
x_best_cost = x_total_costs[total_costs.index(min(total_costs))]
x_total_costs = np.arange(0, len(total_costs), dtype=int)
plt.plot(x_total_costs,total_costs, label="Costos totales")
plt.scatter(x_best_cost,best_cost, color="red", label="Mejor costo")

stop = timeit.default_timer()
runtime = stop - start

#print solution
print("#######################################################")
if (best_cost > fcost):
  print(f'Mejor combinación de centros:\n{best_solution_centers}')
  print("Mejor costo: " + str(best_cost))
else:
  print("Solución infactible")
print("Runtime: " + str(runtime))
#if user inputs --plot
if parser.parse_args().plot:
  plt.legend("upper right",fontsize=12)
  plt.xlabel('iteraciones')
  plt.ylabel('costo')
  plt.legend()
  plt.title(file.split('/')[1])
  if(best_cost <= fcost ):
    best_cost = "Solución Infactible"
    fig = plt.figure(figsize=(5, 1.5))
    text = fig.text(0.5, 0.5, 'Oh no!\nSolución infactible para\n esta instancia.\n:(', ha='center', va='center', size=20)
    text.set_path_effects([path_effects.Normal()])
  plt.show()

