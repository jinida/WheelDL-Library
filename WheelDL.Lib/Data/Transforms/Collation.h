#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

#include "../Dataset/BaseDataset.h" 
#include "../Dataset/PredDataset.h"

namespace WheelDL {
    namespace Data {
        struct DataExampleCollation : public torch::data::transforms::Collation<WheelDL::Data::Dataset::DataExample>
        {
            WheelDL::Data::Dataset::DataExample apply_batch(std::vector<WheelDL::Data::Dataset::DataExample> examples) override;
        };

        struct PredDataExampleCollation : public torch::data::transforms::Collation<WheelDL::Data::Dataset::PredDataExample>
        {
            WheelDL::Data::Dataset::PredDataExample apply_batch(std::vector<WheelDL::Data::Dataset::PredDataExample> examples) override;
		};
    } // namespace Data
} // namespace WheelDL