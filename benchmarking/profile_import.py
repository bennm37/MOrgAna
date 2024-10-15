import cProfile
import pstats 

with cProfile.Profile() as pr:
    import morgana

ps = pstats.Stats(pr)
ps.sort_stats("tottime")
ps.dump_stats("stats.dmp")