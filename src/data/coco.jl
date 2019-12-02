module COCO

# using ZipFile
using JSON
using ..Data: deps

const dir = joinpath(@__DIR__, "../../deps/coco")

function load_images(zipname, name, N)
  if isfile(deps("coco/images/$zipname.zip"))
    @info "COCO $zipname.zip found"
      if isdir(deps("coco/images/$name"))
        @info "COCO $name found"
        # TODO: load N images
	
      else
	@info "COCO $name not found, extracting zip file"
	# TODO: Extract zip file
	# unzipped_file = ZipFile.Reader(deps("coco/images/$name.zip"))
      end
  else
	@info "COCO $zipname.zip not found"
  end
end

function load_labels(zipname, name, N)
  if isfile(deps("coco/labels/$zipname.zip"))
    @info "COCO $zipname.zip found"
      if isdir(deps("coco/labels/$name"))
        @info "COCO $name found"
        # TODO: load N labels
	data = JSON.parsefile(deps("coco/labels/$name.json"))
      else
	@info "COCO $name not found, extracting zip file"
	# TODO: Extract zip file
	# unzipped_file = ZipFile.Reader(deps("coco/labels/$name.zip"))
      end
  else
	@info "COCO $zipname.zip not found"
  end
end

test_images() = load_images("test2017", "test2017", 10)
train_images() = load_images("train2017", "train2017", 100)
val_images() = load_images("val2017", "val2017", 20)
test_labels() = load_labels("image_info_test2017", "test2017", 10)
train_labels() = load_labels("annotations_trainval2017", "train2017", 100)
val_labels() = load_labels("annotations_trainval2017", "val2017", 20)


end
