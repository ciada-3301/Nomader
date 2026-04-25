# Helper to parse useful system stats from the Raspberry pi.

"""
Functions = system[get_cpu_temp,
                   get_cpu_volts,
                   get_ram_usage,
                   get_cpu_usage,
                   get_throttled_status
                    ]
"""

import os
import psutil
import time

class system:
    @staticmethod
    def get_cpu_temp():
        # Uses vcgencmd to get core temperature
        temp = os.popen("vcgencmd measure_temp").readline()
        return temp.replace("temp=", "").replace("'C\n", "")

    @staticmethod
    def get_cpu_volts():
        # Uses vcgencmd to get core voltage
        volts = os.popen("vcgencmd measure_volts core").readline()
        return volts.replace("volt=", "").replace("V\n", "")

    @staticmethod
    def get_ram_usage():
        # Returns (percent, used_mb, total_mb)
        ram = psutil.virtual_memory()
        return ram.percent, ram.used // 1048576, ram.total // 1048576

    @staticmethod
    def get_cpu_usage():
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    def get_throttled_status():
        # Returns the hex status; 0x0 means everything is fine.
        # Common flags: 0x50005 (Currently under-voltage & throttled)
        status = os.popen("vcgencmd get_throttled").readline()
        return status.replace("throttled=", "").strip()



