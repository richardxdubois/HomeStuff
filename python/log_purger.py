import os
import shutil
import datetime
import glob


class log_purger():

    def __init__(self, config_string):
        """
        Args:
            config_string (str): "path_pattern:max_age_days:max_size_gb"
                e.g. "~/Library/Logs/JetBrains:7:1.0"
        """
        parts = config_string.split(":")
        self.base_path = os.path.expanduser(parts[0])
        self.max_age_days = int(parts[1])
        self.max_size_gb = float(parts[2])

    def check(self):
        rc = 0
        msg = "OK"

        try:
            if not os.path.exists(self.base_path):
                return 0, f"Path {self.base_path} not found — nothing to purge"

            total_purged_bytes = 0
            total_purged_items = 0
            cutoff = datetime.datetime.now().timestamp() - (
                self.max_age_days * 86400
            )

            # Purge threadDump directories (always)
            for dirpath in glob.glob(
                os.path.join(self.base_path, "**/threadDumps*"),
                recursive=True
            ):
                size = self._dir_size(dirpath)
                shutil.rmtree(dirpath, ignore_errors=True)
                total_purged_bytes += size
                total_purged_items += 1
                print(f"  Purged: {dirpath} "
                      f"({size / (1024**2):.1f} MB)")

            # Purge old log files beyond max_age_days
            for dirpath, dirnames, filenames in os.walk(self.base_path):
                for f in filenames:
                    fpath = os.path.join(dirpath, f)
                    try:
                        if os.path.getmtime(fpath) < cutoff:
                            size = os.path.getsize(fpath)
                            os.remove(fpath)
                            total_purged_bytes += size
                            total_purged_items += 1
                    except OSError:
                        pass

            # Check remaining size against max
            remaining = self._dir_size(self.base_path)
            remaining_gb = remaining / (1024 ** 3)
            purged_mb = total_purged_bytes / (1024 ** 2)

            print(datetime.datetime.now(),
                  f" Purged {total_purged_items} items"
                  f" ({purged_mb:.1f} MB)."
                  f" Remaining: {remaining_gb:.2f} GB")

            if remaining_gb > self.max_size_gb:
                msg = (f"WARNING: {self.base_path} still"
                       f" {remaining_gb:.1f} GB"
                       f" (limit: {self.max_size_gb} GB)."
                       f" Purged {purged_mb:.1f} MB")
                rc = 1
            else:
                msg = (f"OK - Purged {purged_mb:.1f} MB,"
                       f" {remaining_gb:.2f} GB remaining")
                rc = 0

        except Exception as e:
            msg = f"Purge error: {e}"
            rc = 1
            print(datetime.datetime.now(), f" {msg}")

        return rc, msg

    def _dir_size(self, path):
        """Get total size of directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total


if __name__ == "__main__":
    p = log_purger("~/Library/Logs/JetBrains:7:1.0")
    rc, msg = p.check()
    print(f"rc={rc}, msg={msg}")
