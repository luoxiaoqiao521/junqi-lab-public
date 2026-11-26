"""Run action-level nomination as a distinct, predeclared follow-up."""
import sys

from serial_dark_heritage_nominations import main
from serial_dynamic_reply import require


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected action-level nomination stage")
    main(bool(sys.argv[1:]),action_level=True)
