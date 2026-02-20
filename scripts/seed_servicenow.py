#!/usr/bin/env python3
"""
Seed ServiceNow dev instance with demo data.
Creates sample incidents and KB articles for the hackathon demo.

Usage: python scripts/seed_servicenow.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

INSTANCE = os.environ.get("SERVICENOW_INSTANCE_URL", "").rstrip("/")
USER = os.environ.get("SERVICENOW_USERNAME", "admin")
PASS = os.environ.get("SERVICENOW_PASSWORD", "")

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

INCIDENTS = [
    {
        "short_description": "High CPU utilization on web-prod-03",
        "description": (
            "CPU at 98% on web-prod-03 after deployment v2.4.1 "
            "(2 hours ago). Response times degraded from 200ms to 3s. "
            "Auto-scaling has not triggered."
        ),
        "urgency": "1",
        "impact": "1",
        "category": "Software",
        "subcategory": "Operating System",
    },
    {
        "short_description": "Memory leak in api-gateway-01",
        "description": (
            "Heap at 94% and growing steadily on api-gateway-01. "
            "Connection pool exhaustion warnings in application logs. "
            "JVM GC pauses increasing."
        ),
        "urgency": "2",
        "impact": "2",
        "category": "Software",
        "subcategory": "Operating System",
    },
    {
        "short_description": "Disk space critical at 95% on db-replica-02",
        "description": (
            "Disk at 95% on db-replica-02. Log files consuming 40GB. "
            "Database temp files growing unbounded. Risk of "
            "database crash if 100% reached."
        ),
        "urgency": "1",
        "impact": "2",
        "category": "Hardware",
        "subcategory": "Disk",
    },
    {
        "short_description": "Network latency spike between app and db tiers",
        "description": (
            "Latency spiked to 500ms between app-tier and db-tier. "
            "Packet loss at 2%. Started after scheduled network "
            "maintenance window at 02:00 UTC."
        ),
        "urgency": "2",
        "impact": "1",
        "category": "Network",
        "subcategory": "IP Address",
    },
    {
        "short_description": "SSL cert expiring in 48h for *.prod.company.com",
        "description": (
            "SSL certificate for *.prod.company.com expires in 48 hours. "
            "Auto-renewal via Let's Encrypt failed — ACME challenge "
            "returning 404. Manual intervention required."
        ),
        "urgency": "2",
        "impact": "2",
        "category": "Software",
        "subcategory": "Operating System",
    },
]

KB_ARTICLES = [
    {
        "short_description": "Runbook: High CPU Troubleshooting",
        "text": (
            "1. Run `top -b -n1 | head -20` to identify top CPU consumers\n"
            "2. Check `journalctl -u <service> --since '2 hours ago'`\n"
            "3. If deployment-related: `kubectl rollout undo deployment/<name>`\n"
            "4. Check auto-scaling: `kubectl get hpa`\n"
            "5. If process runaway: `kill -SIGTERM <pid>` then restart\n"
            "6. Monitor for 15 min: `watch -n5 'uptime; free -h'`\n"
            "7. Escalate to P1 bridge if CPU > 95% for 30+ min"
        ),
        "workflow_state": "published",
    },
    {
        "short_description": "Runbook: Memory Leak Investigation",
        "text": (
            "1. Capture heap dump: `jmap -dump:live,format=b,file=/tmp/heap.hprof <pid>`\n"
            "2. Check connection pools: `netstat -an | grep ESTABLISHED | wc -l`\n"
            "3. Review GC logs: `jstat -gc <pid> 1000 10`\n"
            "4. Compare with baseline memory profile\n"
            "5. If conn pool leak: restart with `-Dpool.maxActive=50`\n"
            "6. Schedule maintenance restart during low-traffic window\n"
            "7. File bug for development team with heap analysis"
        ),
        "workflow_state": "published",
    },
    {
        "short_description": "Runbook: Disk Space Recovery",
        "text": (
            "1. Identify top consumers: `du -sh /* | sort -rh | head -20`\n"
            "2. Check log rotation: `cat /etc/logrotate.d/*`\n"
            "3. Compress old logs: `find /var/log -name '*.log.*' -mtime +7 -exec gzip {} \\;`\n"
            "4. Clear app caches: check `/tmp`, `/var/cache`\n"
            "5. Remove orphaned DB temp files\n"
            "6. Update logrotate: keep max 7 days, compress daily\n"
            "7. Set alert threshold at 80% to prevent recurrence"
        ),
        "workflow_state": "published",
    },
]


def create_record(table, data):
    """Create a record in ServiceNow."""
    url = f"{INSTANCE}/api/now/table/{table}"
    resp = requests.post(url, auth=(USER, PASS), headers=HEADERS, json=data)
    if resp.status_code == 201:
        result = resp.json()["result"]
        print(f"  ✓ Created {table}: {data['short_description'][:50]} → {result['sys_id'][:8]}")
        return result
    else:
        print(f"  ✗ Failed {table}: {resp.status_code} {resp.text[:100]}")
        return None


def main():
    if not INSTANCE or not PASS:
        print("Set SERVICENOW_INSTANCE_URL and SERVICENOW_PASSWORD in .env")
        return

    print(f"Seeding {INSTANCE}...")
    print(f"\nCreating {len(INCIDENTS)} incidents...")
    for inc in INCIDENTS:
        create_record("incident", inc)

    print(f"\nCreating {len(KB_ARTICLES)} KB articles...")
    for kb in KB_ARTICLES:
        create_record("kb_knowledge", kb)

    print("\n✓ Seeding complete!")


if __name__ == "__main__":
    main()
