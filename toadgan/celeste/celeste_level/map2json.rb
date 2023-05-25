require_relative  "celeste_map"
# fn = 'app/Content/Maps/1-ForsakenCity.bin'

a = CelesteMap.new(ARGV[0])
a.write_json(ARGV[1])

