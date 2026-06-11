#!/usr/bin/env python3
"""
EX8000 Graceful Reboot Manager
Reboots Netgear EX8000 devices via API with smart plug fallback
"""

import yaml
import requests
from requests.auth import HTTPBasicAuth
import subprocess
import time
from datetime import datetime
from pathlib import Path
import logging
import sys


class EX8000RebootManager:
    """
    Manages graceful reboots of Netgear EX8000 access points
    """

    def __init__(self, config_file='ex8000_config.yaml'):
        """
        Initialize with configuration from YAML file

        Args:
            config_file: Path to YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self):
        """Load and validate configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}\n"
                f"Please create it based on ex8000_config.yaml template"
            )

        # Check file permissions (should be 600)
        if self.config_file.stat().st_mode & 0o777 != 0o600:
            print(f"⚠️  Warning: {self.config_file} should have 600 permissions")
            print(f"   Run: chmod 600 {self.config_file}")

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        if 'devices' not in config:
            raise ValueError("Config file must contain 'devices' section")

        return config

    def _setup_logging(self):
        """Configure logging based on config file"""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('log_file', 'ex8000_reboot.log')

        # Create log directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def ping(self, ip, count=None, timeout=None):
        """
        Test if device is reachable via ping

        Args:
            ip: IP address to ping
            count: Number of ping attempts (from config if not specified)
            timeout: Timeout in seconds (from config if not specified)

        Returns:
            bool: True if device responds, False otherwise
        """
        if count is None:
            count = self.config.get('health_check', {}).get('ping_count', 2)
        if timeout is None:
            timeout = self.config.get('health_check', {}).get('ping_timeout', 2)

        try:
            result = subprocess.run(
                ['ping', '-c', str(count), '-W', str(timeout), ip],
                capture_output=True,
                timeout=timeout * count + 2
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False

    def api_reboot(self, device_config):
        """
        Attempt graceful reboot via device API
        Tries SOAP first, then falls back to HTTP endpoints
        """
        name = device_config['name']
        ip = device_config['ip']

        self.logger.info(f"Attempting API reboot of {name} at {ip}")

        # Try SOAP first (most reliable for Netgear)
        if self.soap_reboot(device_config):
            return True

        # Fallback to HTTP endpoints
        self.logger.info("SOAP failed, trying HTTP endpoints...")

        username = device_config['username']
        password = device_config['password']

        attempts = [
            ('GET', f"http://{ip}/reboot.cgi", None),
            ('POST', f"http://{ip}/apply.cgi",
             {'submit_flag': 'reboot', 'submit_button': 'Reboot'}),
            ('GET', f"http://{ip}/currentsetting.htm",
             {'action': 'reboot'}),
        ]

        for method, url, params in attempts:
            try:
                self.logger.info(f"Trying: {method} {url}")

                if method == 'GET':
                    response = requests.get(
                        url,
                        params=params,
                        auth=HTTPBasicAuth(username, password),
                        timeout=10
                    )
                else:
                    response = requests.post(
                        url,
                        data=params,
                        auth=HTTPBasicAuth(username, password),
                        timeout=10
                    )

                if response.status_code == 200:
                    time.sleep(5)
                    if not self.ping(ip, count=1):
                        self.logger.info(f"✓ Device rebooting")
                        return True

            except requests.exceptions.Timeout:
                return True
            except Exception as e:
                self.logger.debug(f"Failed: {e}")
                continue

        return False

    def soap_reboot(self, device_config):
        """
        Attempt reboot via Netgear SOAP API

        Args:
            device_config: Device configuration dict from YAML

        Returns:
            bool: True if reboot command succeeded, False otherwise
        """
        name = device_config['name']
        ip = device_config['ip']
        username = device_config['username']
        password = device_config['password']

        self.logger.info(f"Attempting SOAP API reboot of {name} at {ip}")

        soap_url = f"http://{ip}:80/soap/server_sa/"

        # SOAP envelope for reboot
        soap_body = """<?xml version="1.0" encoding="utf-8" standalone="no"?>
    <SOAP-ENV:Envelope 
      xmlns:SOAPSDK1="http://www.w3.org/2001/XMLSchema" 
      xmlns:SOAPSDK2="http://www.w3.org/2001/XMLSchema-instance" 
      xmlns:SOAPSDK3="http://schemas.xmlsoap.org/soap/encoding/" 
      xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/">
    <SOAP-ENV:Header>
    </SOAP-ENV:Header>
    <SOAP-ENV:Body>
    <M1:Reboot xmlns:M1="urn:NETGEAR-ROUTER:service:DeviceConfig:1">
    </M1:Reboot>
    </SOAP-ENV:Body>
    </SOAP-ENV:Envelope>"""

        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': '"urn:NETGEAR-ROUTER:service:DeviceConfig:1#Reboot"'
        }

        try:
            response = requests.post(
                soap_url,
                data=soap_body,
                headers=headers,
                auth=HTTPBasicAuth(username, password),
                timeout=10
            )

            self.logger.info(f"SOAP response status: {response.status_code}")
            self.logger.debug(f"SOAP response: {response.text[:500]}")

            if response.status_code == 200:
                # Check if device goes down
                time.sleep(5)
                if not self.ping(ip, count=1):
                    self.logger.info(f"✓ Device is rebooting (SOAP successful)")
                    return True
                else:
                    self.logger.warning(f"SOAP returned 200 but device still responding")

        except requests.exceptions.Timeout:
            self.logger.info("SOAP timeout - device may be rebooting")
            return True

        except Exception as e:
            self.logger.error(f"SOAP reboot failed: {e}")

        return False

    def verify_reboot(self, device_config, wait_time=None):
        """
        Verify device rebooted by checking if it goes down then comes back up

        Args:
            device_config: Device configuration dict
            wait_time: Seconds to wait (from config if not specified)

        Returns:
            bool: True if device successfully rebooted, False otherwise
        """
        name = device_config['name']
        ip = device_config['ip']

        if wait_time is None:
            wait_time = self.config.get('health_check', {}).get(
                'post_reboot_wait', 120
            )

        self.logger.info(f"Verifying reboot of {name}...")

        # Wait a bit for device to start rebooting
        time.sleep(10)

        # Check if device is down (rebooting)
        if self.ping(ip):
            self.logger.warning(
                f"⚠️  {name} still responding immediately after reboot command"
            )

        # Wait for device to come back up
        self.logger.info(f"Waiting {wait_time}s for {name} to reboot...")
        time.sleep(wait_time)

        # Verify it's back up
        if self.ping(ip):
            self.logger.info(f"✓ {name} is back online")
            return True
        else:
            self.logger.error(f"✗ {name} did not come back online")
            return False

    def reboot_device(self, device_key, verify=True):
        """
        Reboot a specific device by key from config

        Args:
            device_key: Key name from YAML devices section
            verify: Whether to verify reboot succeeded

        Returns:
            bool: True if reboot succeeded, False otherwise
        """
        if device_key not in self.config['devices']:
            self.logger.error(f"Device '{device_key}' not found in config")
            return False

        device_config = self.config['devices'][device_key]
        name = device_config['name']

        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Starting reboot of {name}")
        self.logger.info(f"{'=' * 60}")

        # Check if device is reachable before reboot
        if not self.ping(device_config['ip']):
            self.logger.warning(f"⚠️  {name} not reachable - may already be down")

        # Attempt API reboot
        success = self.api_reboot(device_config)

        if not success:
            self.logger.error(f"API reboot failed for {name}")
            return False

        # Optionally verify reboot
        if verify:
            return self.verify_reboot(device_config)

        return True

    def reboot_all_devices(self, delay_between=60):
        """
        Reboot all configured devices

        Args:
            delay_between: Seconds to wait between device reboots

        Returns:
            dict: Results for each device
        """
        results = {}
        devices = list(self.config['devices'].items())

        for i, (device_key, device_config) in enumerate(devices):
            results[device_key] = self.reboot_device(device_key)

            # Delay between devices (except after last one)
            if i < len(devices) - 1:
                self.logger.info(f"Waiting {delay_between}s before next device...")
                time.sleep(delay_between)

        return results

    def status_report(self):
        """
        Generate status report of all configured devices

        Returns:
            dict: Status of each device
        """
        self.logger.info("Checking status of all devices...")

        status = {}
        for device_key, device_config in self.config['devices'].items():
            reachable = self.ping(device_config['ip'])
            status[device_key] = {
                'name': device_config['name'],
                'ip': device_config['ip'],
                'reachable': reachable,
                'timestamp': datetime.now().isoformat()
            }

            icon = "✓" if reachable else "✗"
            self.logger.info(
                f"{icon} {device_config['name']:20s} - "
                f"{'ONLINE' if reachable else 'OFFLINE'}"
            )

        return status


def main():
    """Command-line interface for EX8000 reboot manager"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Manage Netgear EX8000 reboots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status of all devices
  python ex8000_reboot.py --status

  # Reboot specific device
  python ex8000_reboot.py --reboot bedroom_ex8000

  # Reboot all devices
  python ex8000_reboot.py --reboot-all

  # Use custom config file
  python ex8000_reboot.py --config /path/to/config.yaml --status
        """
    )

    parser.add_argument(
        '--config',
        default='ex8000_config.yaml',
        help='Path to YAML configuration file (default: ex8000_config.yaml)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check and report status of all devices'
    )

    parser.add_argument(
        '--reboot',
        metavar='DEVICE',
        help='Reboot specific device by key (e.g., bedroom_ex8000)'
    )

    parser.add_argument(
        '--reboot-all',
        action='store_true',
        help='Reboot all configured devices'
    )

    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification that device came back online'
    )

    args = parser.parse_args()

    # Initialize manager
    try:
        manager = EX8000RebootManager(config_file=args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Execute requested action
    if args.status:
        manager.status_report()

    elif args.reboot:
        success = manager.reboot_device(
            args.reboot,
            verify=not args.no_verify
        )
        sys.exit(0 if success else 1)

    elif args.reboot_all:
        results = manager.reboot_all_devices()
        all_success = all(results.values())
        sys.exit(0 if all_success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
