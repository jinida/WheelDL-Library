#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

#include "WheelDL.Lib/Data/Dataset/BaseDataset.h" 

namespace WheelDL {
    namespace Data {
        struct DataExampleCollation : public torch::data::transforms::Collation<WheelDL::Data::Dataset::DataExample>
        {
            WheelDL::Data::Dataset::DataExample apply_batch(std::vector<WheelDL::Data::Dataset::DataExample> examples) override;
        };

    } // namespace Data
} // namespace WheelDL