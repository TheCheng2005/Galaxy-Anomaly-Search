"""Read either score-CSV header spelling, skipping token-usage footer lines."""
import csv


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = [column.lower() for column in next(reader)]
        return [dict(zip(header, row)) for row in reader
                if row and not row[0].startswith("#")]
