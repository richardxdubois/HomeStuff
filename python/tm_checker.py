import subprocess
import re
import datetime


class tm_checker():

    def __init__(self, max_age_hours=24):
        """
        Args:
            max_age_hours (float): Flag failure if the most recent completed
                backup is older than this. Default 24h.
        """
        self.max_age_hours = max_age_hours

    def _run(self, args):
        """Run a tmutil subcommand, return (stdout, None) or (None, error)."""
        try:
            out = subprocess.check_output(
                ["/usr/bin/tmutil"] + args,
                text=True,
                stderr=subprocess.STDOUT,
            )
            return out, None
        except subprocess.CalledProcessError as e:
            return None, e.output or str(e)
        except FileNotFoundError:
            return None, "tmutil not found"

    def destination_ok(self):
        """
        Check the local/USB destination is configured and mounted.

        Returns:
            (bool, str): (ok, detail)
        """
        out, err = self._run(["destinationinfo"])
        if err is not None:
            return False, f"destinationinfo error: {err.strip()}"

        # destinationinfo lists 'Mount Point' only when the disk is mounted.
        if "Mount Point" in out:
            mp = re.search(r"Mount Point\s*:\s*(.+)", out)
            return True, f"mounted at {mp.group(1).strip()}" if mp else "mounted"

        if re.search(r"Name\s*:", out):
            return False, "destination configured but not mounted"

        return False, "no destination configured"

    def backup_fresh(self):
        """
        Check the most recent completed backup is within max_age_hours.

        Returns:
            (bool, str): (ok, detail)
        """
        out, err = self._run(["latestbackup"])
        if err is not None:
            # latestbackup commonly fails with a permissions message when
            # not run as root.
            if "privilege" in err.lower() or "permission" in err.lower():
                return False, "latestbackup requires elevated privileges"
            return False, f"latestbackup error: {err.strip()}"

        path = out.strip()
        if not path:
            return False, "no completed backups found"

        # Backup dirs are named YYYY-MM-DD-HHMMSS(.backup)
        m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", path)
        if not m:
            return False, f"could not parse backup timestamp from {path}"

        ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d-%H%M%S")
        age_h = (datetime.datetime.now() - ts).total_seconds() / 3600.0

        if age_h > self.max_age_hours:
            return False, f"last backup {age_h:.1f}h old (> {self.max_age_hours}h)"
        return True, f"last backup {age_h:.1f}h old"

    def check(self):

        rc = 1
        msg = "unsuccessful backup check"

        try:
            dest_ok, dest_detail = self.destination_ok()
            fresh_ok, fresh_detail = self.backup_fresh()

            now = datetime.datetime.now()
            print(now, f"TM destination: {dest_detail}")
            print(now, f"TM freshness:   {fresh_detail}")

            if dest_ok and fresh_ok:
                msg = "successful backup check"
                rc = 0
            else:
                failed = []
                if not dest_ok:
                    failed.append(f"destination ({dest_detail})")
                if not fresh_ok:
                    failed.append(f"staleness ({fresh_detail})")
                msg = "backup check failed: " + "; ".join(failed)
                rc = 1

        except Exception as e:
            print(datetime.datetime.now(), f"TM checker Error: {e}")
            rc = 1

        return rc, msg


if __name__ == "__main__":

    c = tm_checker(max_age_hours=24)
    print(c.check())
