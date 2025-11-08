#include "pch.h"
#include "Collation.h"

namespace WheelDL {
	namespace Data {

		WheelDL::Data::Dataset::DataExample DataExampleCollation::apply_batch(std::vector<WheelDL::Data::Dataset::DataExample> examples)
		{
			std::vector<torch::Tensor> dataTensors;
			std::vector<torch::Tensor> classesTensors;
			std::vector<torch::Tensor> targetsTensors;

			unsigned int batchIdx = 0;
			for (auto& example : examples)
			{
				dataTensors.push_back(std::move(example.data));
				classesTensors.push_back(std::move(example.classes));
				targetsTensors.push_back(std::move(example.targets));
			}

			torch::Tensor stackedData = torch::stack(dataTensors, 0);
			torch::Tensor stackedClasses = torch::stack(classesTensors, 0);

			if (targetsTensors.size() == 0)
			{
				return WheelDL::Data::Dataset::DataExample{
					std::move(stackedData),
					std::move(stackedClasses),
					torch::Tensor(),
					torch::Tensor()
				};
			}
			else if (targetsTensors[0].dim() == 3)
			{
				torch::Tensor stackedTargets = torch::stack(targetsTensors, 0);

				return WheelDL::Data::Dataset::DataExample{
					std::move(stackedData),
					std::move(stackedClasses),
					std::move(stackedTargets),
					torch::Tensor()
				};
			}
			else 
			{
				std::vector<torch::Tensor> batchIndices;
				for (size_t i = 0; i != targetsTensors.size(); ++i)
				{
					batchIdx = static_cast<unsigned int>(i);
					torch::Tensor indices = torch::full({ targetsTensors[i].size(0) }, static_cast<float>(batchIdx), torch::kLong);
					batchIndices.push_back(indices);
				}

				torch::Tensor concatenatedTargets = torch::cat(targetsTensors, 0);
				torch::Tensor concatenatedBatchIndices = torch::cat(batchIndices, 0);

				return WheelDL::Data::Dataset::DataExample{
					std::move(stackedData),
					std::move(stackedClasses),
					std::move(concatenatedTargets),
					std::move(concatenatedBatchIndices)
				};
			}
		}

	} // namespace Data
} // namespace WheelDL