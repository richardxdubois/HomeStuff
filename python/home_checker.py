import yaml
import importlib
import logging
import asyncio
from send_gmail import send_gmail
import datetime
from speedtest_dashboard_generator import DashboardGenerator
import time
from disk_dashboard import DiskDashboard

class HomeChecker():
    """
    Check services/connectivity and optionally alert on failures via email (gmail)

    Configured via a yaml file which defines the list of devices to check, the plugin to use and any
    parameter needed to run the plugin.

    Device list in yaml is a map of device names vs [ plugin, arg-string ] (2-element list)

    Current plugins are to ping (pinger), and check the Litter Robot (access_LR4) and Hubitat controller
    (access_hubitat).

    Launchd runs the checker periodically via a plist file. There is a bash script that wraps the checker, setting up
    conda and the specific env needed. It must be run as root to avoid the local network permissions:

    sudo launchctl load /Library/LaunchDaemons/home_checker.plist  (unload to stop it)
    (master copy in ~/Library/LaunchAgentss/home_checker.plist; do a sudo copy to LaunchDaemons/)

    The checker outputs status to a log file and stdout is redirected to a separate file
    """
    def __init__(self, app_config):

        with open(app_config, "r") as f:
            data = yaml.safe_load(f)

        self.logfile = data["log_file"]
        self.email = data["email"]
        devs_2_check = data["devs_2_check"]
        self.router_IP = data["router_IP"]
        self.retries = data["retries"]
        self.dashboard_config = data["dashboard_config"]

        self.logger = self.setup_logger(log_file=self.logfile)

        self.gm = send_gmail(data["gmail_creds"])

        # see if the router is up

        print("first router check", datetime.datetime.now())
        try_router = self.invoke_class_by_name("pinger", "pinger", self.router_IP)
        rc, msg = try_router.check()
        if rc == 1:
            rc, msg = try_router.check()
            print("router ping retry", datetime.datetime.now())

            if rc == 1:
                log_msg = f"router: {msg}"
                self.logger.info(log_msg)
                print("abandon ship on router ping retry", datetime.datetime.now())
                exit()

        print("router ok", datetime.datetime.now())

        self.apps_list = []
        self.devs_list = []
        for dev in devs_2_check:
            app = devs_2_check[dev][0]
            arg = devs_2_check[dev][1]
            print("start init: ", dev, datetime.datetime.now())
            instance = self.invoke_class_by_name(app, app, arg)
            print("after init: ", dev, datetime.datetime.now())
            self.apps_list.append(instance)
            self.devs_list.append(dev)

        print("done with init: ", datetime.datetime.now())

        email_msg = ""
        for i, app in enumerate(self.apps_list):
            a_sync = devs_2_check[self.devs_list[i]][2]
            if a_sync:
                rc, msg = asyncio.run(app.check())
            else:
                rc, msg = app.check()

            if rc == 1:
                for r in range(self.retries):
                    print(f"retrying {self.devs_list[i]}")
                    time.sleep(5)
                    if a_sync:
                        rc, msg = asyncio.run(app.check())
                    else:
                        rc, msg = app.check()
                    if rc == 0:
                        break

            log_msg = f"{self.devs_list[i]}: {msg}"
            self.logger.info(log_msg)
            if rc != 0:
                email_msg += f"\n{log_msg}"

        print("done with check")

        if self.email and email_msg != "":
            self.gm.send("Home systems error report", email_msg)

        dashboard = DashboardGenerator(app_config=self.dashboard_config)
        # Generate the dashboard file
        dashboard.generate_dashboard()

        disk_dashboard = DiskDashboard(
            log_file="/Users/richarddubois/Code/Home/home_checker/disk_space.csv",
            output_file_path="/Users/richarddubois/Code/Home/home_checker/disk_dashboard.html",
            table_days=4
        )
        disk_dashboard.generate()
        print("dashboards generated \n")

    def invoke_class_by_name(self, module_name, class_name, *args, **kwargs):
        """Invokes a class by its name (string) if defined in a different module."""
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)

            # Get the class object from the module
            cls = getattr(module, class_name)

            # Instantiate the class
            instance = cls(*args, **kwargs)
            return instance
        except ImportError:
            print(f"Module '{module_name}' not found.")
            return None
        except AttributeError:
            print(f"Class '{class_name}' not found in module '{module_name}'.")
            return None
        except Exception as e:
            print(f"Error invoking class '{class_name}' in module '{module_name}': {e}")
            return None

    def setup_logger(self, log_file, level=logging.INFO):
        """Sets up a logger that writes to a file and the console."""
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger


if __name__ == "__main__":

    c = HomeChecker("/Users/richarddubois/Code/Home/home_checker/home_checker.yaml")
