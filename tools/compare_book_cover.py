"""Run the book-inspired allied support interruption comparison."""
import sys

from compare_book_candidate import main


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"],support=True)
