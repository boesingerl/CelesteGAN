require_relative  "celeste_map"

a = CelesteMap.new(ARGV[0], fmt: :json)
a.write ARGV[1]

