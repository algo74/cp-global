
from sys import stdout
import numpy as np
import pandas as pd
from docplex.cp.model import CpoModel
import matplotlib.pyplot as plt
import time
import traceback
import sys

import cplex_context

checkpoint_interval = 10000

# names of columns in swf files
col_names = [
    'job_id',
    'Submit Time',
    'Wait Time',
    'Run Time',
    'Number of Allocated Processors',
    'Average CPU Time Used',
    'Used Memory',
    'Requested Number of Processors',
    'Requested Time',
    'Requested Memory',
    'Status',
    'User ID',
    'Group ID',
    'Executable',
    'Queue Number',
    'Partition Number',
    'Preceding Job Number',  # also is reused to store number of "underpredictions"
    'Think Time'  # also is reused to store predictions
]


def _write_solution_to_file(filename, result, numNodes):
    if result is not None:
        with open(filename, 'w') as f:
            f.write("; MaxProcs: {}\n".format(numNodes))
            result.to_csv(f, sep='\t', index=False, header=False)


def _cp_scheduling_attempt(max_nodes, queue, objective_function='AWF', timelimit=100, verbosity='Normal'):
    """
    NOTE: we are using run times instead of predicted run times
    """
    in_debug = verbosity == 'Normal'
  
    mdl = CpoModel()
    # We will search for a solution in time interval from 0 to max_makespan.
    # We will calculate max_makespan as max("durations of running job") + sum("durations of queued jobs").
    # This is just an initial assignment.
    queued_job_dict = {}  # key: job_id, value: (job, requirements, submit_time, duration, interval)
    interval_list = []
    resource_list = []
    if in_debug:
      print("========================================================================================")
      # print("Scheduling at time {}".format(time))
      # print("{} running jobs and {} waiting jobs".format(len(self.jobs.get_running_jobs()), len(queue)))

    # sort queue by submit time
    squeue = queue.sort_values(by=['Submit Time'])
    # min_makespan is the submission time of the first job in the sorted queue
    # min_makespan = squeue.iloc[0]['Submit Time']
    # max_makespan is the time needed if we run one job at a time
    max_makespan = 0
    for _, job in squeue.iterrows():
      max_makespan = max(max_makespan, job['Submit Time']) + job['Run Time']
    print("Max makespan: {}".format(max_makespan))

    # process queued jobs
    for _, job in squeue.iterrows():
      job_id = job['job_id']
      duration = job['Run Time']
      requirements = job['Requested Number of Processors']
      submit_time = job['Submit Time'] 
      if duration < 0 or requirements < 0 or submit_time < 0:
         print("Error: negative value found in job {} with duration {} and requirements {} and submit time {}".format(
                job_id, duration, requirements, submit_time))
      max_start = max_makespan - duration
      # print("Job {} with duration {} and requirements {} can start between {} and {}".format(
      #       job_id, duration, requirements, submit_time, max_start))
      interval = mdl.interval_var(start=(submit_time, max_start), size=duration, optional=False,
                                   name='Q{}'.format(job_id))
      interval_list.append(interval)
      resource_list.append(requirements)
      # convert parameters to float to prevent integer overflow
      # duration = float(duration)
      # requirements = float(requirements)
      # submit_time = float(submit_time)
      queued_job_dict[job_id] = (job, requirements, submit_time, duration, interval)
    # # add job order heuristic constraints
    # size_sorted_queue = sorted(
    #   [(job.predicted_run_time, job.num_required_processors, job.submit_time, job.id) for job in sorted_queue])
    # if len(size_sorted_queue) > 2:
    #   for (prev, next) in zip(size_sorted_queue[:-1], size_sorted_queue[1:]):
    #     if prev[0] == next[0] and prev[1] == next[1]:
    #       mdl.add(dcpm.start_before_start(queued_job_dict[prev[3]][1], queued_job_dict[next[3]][1]))
    # add resource constraint
    node_constraint = (mdl.sum([mdl.pulse(j, n) for j, n in zip(interval_list, resource_list)]) <= max_nodes)
    # node_constraint = dcpm.cumul_range(dcpm.sum([dcpm.pulse(j, n) for j, n in zip(interval_list, resource_list)]), 0, self.nodes.max)
    mdl.add(node_constraint)
    # add objective function
    if objective_function == 'AWF':
      # AWF
      # job_contributions = [nodes * duration * (mdl.end_of(interval) - submit)  ## proper AWF
      job_contributions = [nodes * duration * (mdl.start_of(interval) - submit)  ## optimized AWF
               for job, nodes, submit, duration, interval in queued_job_dict.values()]
      objective_var = mdl.sum(job_contributions)
    elif objective_function == 'AF':
      job_contributions = [(mdl.end_of(interval) - submit)
               for job, nodes, submit, duration, interval in queued_job_dict.values()]
      objective_var = mdl.sum(job_contributions)
    elif objective_function == 'BSLD':
      job_contributions = [mdl.max(1, (mdl.end_of(interval) - submit) / float(max(10, duration)))
               for job, nodes, submit, duration, interval in queued_job_dict.values()]
      objective_var = mdl.sum(job_contributions)
    elif objective_function == 'P2SF':
      # M_job = n * (F ** (p + 1) - Tw ** (p + 1))
      M2 = []
      M3 = []
      for job, nodes, submit, duration, interval in queued_job_dict.values():
        Tw = mdl.start_of(interval) - float(submit)
        # F = mdl.end_of(interval) - float(submit)
        F = Tw + float(duration)
        M2.append(nodes * (F ** 3 - Tw ** 3))
        M3.append(nodes * (F ** 4 - Tw ** 4))
      objective_var = mdl.sum(M3) / mdl.sum(M2)
    else:
      raise ValueError("Unknown objective function: {}".format(objective_function))
    objective_monitor = mdl.minimize(objective_var)
    mdl.add(objective_monitor)
    # res = mdl.solve(TimeLimit=self.timelimit, LogVerbosity='Normal', SearchType='IterativeDiving')
    # get current clock time
    start_time = time.time()
    checkpoint_file_format = objective_function + "_" + str(int(start_time)) + "_ckpt_{}.swf"
    checkpoint_time = start_time + checkpoint_interval
    solver = mdl.start_search(TimeLimit=timelimit, 
                              LogVerbosity=verbosity,
                              RelativeOptimalityTolerance=0, 
                              Presolve='Off',
                              agent=cplex_context.CPLEX_AGENT, 
                              execfile=cplex_context.CPLEX_EXECUTABLE_PATH)
    res = None
    # catch exceptions and save the last solution
    try:
      for solution in solver:
        res = squeue
        # combine the results into a dataframe
        for job_id, (job, requirements, submit_time, duration, interval) in queued_job_dict.items():
          job['Wait Time'] = solution.get_var_solution(interval).get_start() - submit_time
          now = time.time()
          if now - start_time > timelimit:
            print("====== Time limit exceeded ======")
            break
          if checkpoint_time < now:
            print("--------- Checkpointing ---------")
            checkpoint_time = now + checkpoint_interval
            _write_solution_to_file(checkpoint_file_format.format(int(now)), squeue, max_nodes)
      print("====== Time limit reached =======")
    except Exception as e:
      print("====== Exiting due to error =======")
      print("Exception: {}".format(e))
      # print exeption message
      print("".join(traceback.format_exception(*sys.exc_info())))
      # print stack trace
      # print("".join(traceback.format_stack()))
      print("===================================")
    if res is None: 
       print("No solution found")
    return res 


def offline_scheduling(in_swf_filename, out_swf_filename):
    """
    Solve the offline scheduling problem.
    :param swf_filename: the filename of the swf file
    :return: the solution
    """
    numNodes = None

    ## read swf file
    df = pd.read_csv(in_swf_filename, sep='\s+', comment=';',
                     header=None, names=col_names)
    # print(df.head())
    # extracting number of processors
    input_file = open(in_swf_filename, 'r')
    for line in input_file:
        if (line.lstrip().startswith(';')):
            if (line.lstrip().startswith('; MaxProcs:')):
                numNodes = int(line.strip()[11:])
                break
            else:
                continue
        else:
            break
    input_file.close()
    assert numNodes is not None

    print("--------------------------------------------------------------------------------------")
    print("Total number of nodes: {}".format(numNodes))
    max_requested_nodes = df['Requested Number of Processors'].max()
    print("Max job requirements: {} in the following jobs:".format(max_requested_nodes))
    print(df[df['Requested Number of Processors'] == max_requested_nodes])
    print("--------------------------------------------------------------------------------------")

    ## formulate and solve the problem
    result = _cp_scheduling_attempt(numNodes, df, objective_function='AF', timelimit=256500, verbosity='Normal')

    ## save the solution adding "; MaxProcs: <numNodes>" to the header
    _write_solution_to_file(out_swf_filename, result, numNodes)


if __name__ == "__main__":
    in_swf_filename = "data/KTH-SP2.swf"
    out_swf_filename = "results/AF.256500.swf"
    offline_scheduling(in_swf_filename, out_swf_filename)

