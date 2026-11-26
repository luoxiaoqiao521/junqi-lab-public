"""Replay the same bomb alternatives using two unused rollout streams."""
import sys

from compare_bomb_reply_and_reserve import main, require


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"],resample=True)
