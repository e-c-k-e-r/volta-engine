import sys
import store
from report import total

if __name__ == "__main__":
    rows = store.fetch_data(sys.argv[1])
    print(len(rows), total(sys.argv[1]))
