// CPU-only 校验工具：读 PTQW/PTQ2 + PTQC，打印结构计数。
// 用途：09 导出候选后做跨语言校验（不碰 GPU、不需要 libtorch）。
#include <cstdio>
#include <string>
#include <vector>

#include "ptq_model.h"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <model.bin> <calib.ptqc>\n", argv[0]);
        return 2;
    }
    std::string err;
    ptqmodel::Model m;
    if (!m.load(argv[1], &err))
    {
        fprintf(stderr, "模型加载失败: %s\n", err.c_str());
        return 1;
    }
    std::vector<std::pair<float, float>> ranges;
    if (!ptqmodel::load_ptqc(argv[2], m.convs.size(), &ranges, &err))
    {
        fprintf(stderr, "PTQC 校验失败: %s\n", err.c_str());
        return 1;
    }
    printf("模型 %s：n_conv=%zu n_block=%zu stem_pool=%d "
           "grouped=%d depthwise=%d relu6=%d residual2=%d fc=%d->%d\n",
           argv[1], m.convs.size(), m.blocks.size(), m.stem_pool,
           m.n_grouped(), m.n_depthwise(), m.n_relu6(), m.n_residual2(),
           m.fc_in, m.fc_out);
    printf("PTQC %s：n=%zu（文件 8+8n=%zu B）首层 [%g, %g]\n",
           argv[2], ranges.size(), 8 + 8 * ranges.size(),
           ranges.empty() ? 0.0f : ranges.front().first,
           ranges.empty() ? 0.0f : ranges.front().second);
    return 0;
}
