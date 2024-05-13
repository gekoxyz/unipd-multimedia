#!/usr/bin/env python3
from ping3 import ping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

server = "paris.testdebit.info"
ping_tracerute_out_file = "ping_traceroute_out.txt"
traceroute_out_file = "traceroute_out.txt"
rtt_out_file = "rtt_out.csv"

def collect_traceroute_data():
  # collect traceroute data
  with open(traceroute_out_file, "w") as file:
    process = subprocess.run(["traceroute", server], capture_output=True, text=True)
    file.write(str(process.stdout))
    file.close()
  # collect ping data
  ttl = 30
  with open(ping_tracerute_out_file, "w") as file:
    while ttl > 0:
      file.write(f"TTL = {ttl}\n")
      seconds = ping(server, unit="ms", ttl=ttl)
      file.write(f"{seconds}\n")
      ttl -= 1
    file.close()

def process_traceroute_data():
  # process traceroute data
  with open(traceroute_out_file, "r") as file:
    lines = file.readlines()
    last_line = lines[-1]
    first_word = last_line.split()[0]
    print(f"number of steps with traceroute: {first_word}")
    file.close()
  # process ping data
  with open(ping_tracerute_out_file, "r") as file:
    lines = file.readlines()
    ttl_buffer = -1
    for line in lines:
      if line.startswith("TTL = "):
        ttl_value = int(line.split("=")[1].strip())
        ttl_buffer = ttl_value + 1
      elif line == "False\n":
        if ttl_buffer == -1:
          print(f"no valid TTL found")
        else:
          print(f"number of steps with ping: {ttl_buffer}")
          break
    file.close()

# save the ping data in the file 
# the ping payload data must vary between 10 and 1472 bytes and for each
# instance K tests must be ran
def collect_rtt_data():
  # numero di pacchetti spediti
  k = 75
  payload_sizes = np.linspace(10, 1472, 20, dtype = int)
  with open(rtt_out_file, "w") as file:
    file.write("time,size\n")
    for payload_size in payload_sizes:
      for _ in range(k):
        seconds = ping(server, unit='ms', size=payload_size)
        file.write(f"{seconds},{payload_size}\n")
    file.close()

def process_rtt_data():
  with open(rtt_out_file, 'r') as file:
    df = pd.read_csv(file)
    file.close()

    df.plot(kind='scatter', x='size', y='time')
    plt.savefig('rtt_data.png')

    df_clean = df.dropna()
    x = df_clean["size"].values
    y = df_clean["time"].values

    coefs = np.polyfit(x, y, 1)
    print(coefs)
    payload_sizes = np.linspace(10, 1472, 20, dtype = int)
    y = np.polyval(coefs, x)
    plt.plot(x, y, color='purple')
    plt.savefig("rtt_fit.png")

    # coefs = np.polynomial.polynomial.polyfit(df.dropna()[["size"]].values, df.dropna()[["time"]].values, 1)
    poly = np.poly1d(coefs)
    print(poly(1000))  # evaluate the polynomial at x=1000
  return

if __name__ == "__main__":
  # collect_traceroute_data()
  # process_traceroute_data()
  # collect_rtt_data()
  process_rtt_data()

