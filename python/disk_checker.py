import shutil
import datetime
import os


class disk_checker():

    DEFAULT_LOG_FILE = "/Users/richarddubois/Code/Home/home_checker/disk_space.csv"

    def __init__(self, config_string):
        """
        Args:
            config_string (str): "mount_point:threshold_gb:label[:log_file]"
                e.g. "/System/Volumes/Data:20:Macintosh HD"
                e.g. "/Volumes/TM:200:Time Machine:/Volumes/Data/Home/disk_space.csv"
        """
        parts = config_string.split(":")
        self.mount_point = parts[0]
        self.threshold_gb = float(parts[1])
        self.label = parts[2] if len(parts) > 2 else self.mount_point
        self.log_file = parts[3] if len(parts) > 3 else disk_checker.DEFAULT_LOG_FILE

    def check(self):

        rc = 1
        msg = "unsuccessful connection"

        try:
            usage = shutil.disk_usage(self.mount_point)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            used_pct = (usage.used / usage.total) * 100

            print(datetime.datetime.now(),
                  f" {self.label} ({self.mount_point}):"
                  f" {free_gb:.1f} GB free of {total_gb:.1f} GB"
                  f" ({used_pct:.1f}% used)")

            if free_gb >= self.threshold_gb:
                status = "OK"
                msg = (f"OK - {free_gb:.1f} GB free"
                       f" ({used_pct:.1f}% used)")
                rc = 0
            else:
                status = "LOW"
                msg = (f"LOW DISK SPACE - {free_gb:.1f} GB free"
                       f" (threshold: {self.threshold_gb:.0f} GB,"
                       f" {used_pct:.1f}% used)")
                rc = 1

            self._log_entry(status, free_gb, total_gb, used_pct)

        except FileNotFoundError:
            msg = f"Mount point '{self.mount_point}' not found - drive may be unmounted"
            print(datetime.datetime.now(), f" {msg}")
            self._log_entry("UNMOUNTED", 0, 0, 0)

        except Exception as e:
            msg = f"Disk check error: {e}"
            print(datetime.datetime.now(), f" {msg}")
            self._log_entry("ERROR", 0, 0, 0)

        return rc, msg

    def _log_entry(self, status, free_gb, total_gb, used_pct):
        if not self.log_file:
            return

        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        file_exists = os.path.exists(self.log_file)

        # Check if file needs a newline fix
        needs_newline = False
        if file_exists and os.path.getsize(self.log_file) > 0:
            with open(self.log_file, 'rb') as f:
                f.seek(-1, 2)
                needs_newline = f.read(1) != b'\n'

        with open(self.log_file, 'a') as f:
            if not file_exists:
                f.write("timestamp,label,mount_point,status,"
                        "free_gb,total_gb,used_pct,threshold_gb\n")

            if needs_newline:
                f.write('\n')

            f.write(f'{timestamp},"{self.label}",{self.mount_point},'
                    f'{status},{free_gb:.2f},{total_gb:.2f},'
                    f'{used_pct:.1f},{self.threshold_gb:.0f}\n')
            f.flush()
            os.fsync(f.fileno())

        print(f"  Logged to {self.log_file}")


if __name__ == "__main__":

    disks = [
        "/Volumes/Macintosh HD:20:Macintosh HD",
        "/Volumes/Data:20:Data",
        "/Volumes/TM:200:Time Machine",
        "/Volumes/External_SSD_Deb:10:External SSD",
    ]

    for d in disks:
        checker = disk_checker(d)
        rc, msg = checker.check()
        print(f"  rc={rc}, msg={msg}\n")
