import ping3
import numpy as np
# ping3.DEBUG = True
import subprocess

# server = "paris.testdebit.info"
server = "1.1.1.1"
ping_tracerute_out_file = "ping_traceroute_out.txt"
traceroute_output_file = "traceroute_out.txt"
k = 5 # numero di pacchetti spediti
payload_sizes = np.linspace(10, 1450, 10, dtype = int)

def collect_data():
  # collect traceroute data
  with open(traceroute_output_file, "w") as file:
    process = subprocess.run(["traceroute", server], capture_output=True, text=True)
    file.write(str(process.stdout))
    file.close()
  # collect ping data
  ttl = 30
  with open(ping_tracerute_out_file, "w") as file:
    while ttl > 0:
      file.write(f"TTL = {ttl}\n")
      seconds = ping3.ping(server, unit="ms", ttl=ttl)
      file.write(f"{seconds}\n")
      ttl -= 1
    file.close()

def process_data():
  # process traceroute data
  with open(traceroute_output_file, "r") as file:
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

if __name__ == "__main__":
  # collect_data()
  process_data()



  # with open(output_file, "w") as file:
    # for payload_size in payload_sizes:
    #   file.write(f"payload_size={payload_size}\n")
    #   for i in range(k):
    #     seconds = ping3.ping(server, unit='ms', size=payload_size)
    #     file.write(f"{seconds}\n")
