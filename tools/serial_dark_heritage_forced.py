"""Force previously omitted contacts with identical subsequent policies."""
import sys

from serial_dark_heritage_nominations import main
from serial_dynamic_reply import require


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected omitted-capture stage")
    main(bool(sys.argv[1:]),action_level=True,force_contact=True)
