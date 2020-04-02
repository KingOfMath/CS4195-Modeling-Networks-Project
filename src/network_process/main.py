from temporal_network import TemporalNetworkWrapper

# First run: specify source file rel.rating tn = TemporalNetworkWrapper("network.json", "ral.rating")
# if .json doesn't alredy exist in sources dir.
tn = TemporalNetworkWrapper("network.json")

# Example to teneto
tn = tn.to_teneto()
