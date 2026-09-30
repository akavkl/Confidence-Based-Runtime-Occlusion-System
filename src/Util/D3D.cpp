#include "Util/D3D.h"

namespace CBRO::Util
{
	std::string_view FormatName(DXGI_FORMAT a_format) noexcept
	{
		switch (a_format) {
		case DXGI_FORMAT_UNKNOWN:
			return "UNKNOWN"sv;
		case DXGI_FORMAT_R32G32B32A32_FLOAT:
			return "R32G32B32A32_FLOAT"sv;
		case DXGI_FORMAT_R16G16B16A16_FLOAT:
			return "R16G16B16A16_FLOAT"sv;
		case DXGI_FORMAT_R16G16B16A16_UNORM:
			return "R16G16B16A16_UNORM"sv;
		case DXGI_FORMAT_R32G32_FLOAT:
			return "R32G32_FLOAT"sv;
		case DXGI_FORMAT_R32G8X24_TYPELESS:
			return "R32G8X24_TYPELESS"sv;
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
			return "D32_FLOAT_S8X24_UINT"sv;
		case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
			return "R32_FLOAT_X8X24_TYPELESS"sv;
		case DXGI_FORMAT_R10G10B10A2_UNORM:
			return "R10G10B10A2_UNORM"sv;
		case DXGI_FORMAT_R11G11B10_FLOAT:
			return "R11G11B10_FLOAT"sv;
		case DXGI_FORMAT_R8G8B8A8_UNORM:
			return "R8G8B8A8_UNORM"sv;
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
			return "R8G8B8A8_UNORM_SRGB"sv;
		case DXGI_FORMAT_R16G16_FLOAT:
			return "R16G16_FLOAT"sv;
		case DXGI_FORMAT_R16G16_UNORM:
			return "R16G16_UNORM"sv;
		case DXGI_FORMAT_R32_TYPELESS:
			return "R32_TYPELESS"sv;
		case DXGI_FORMAT_D32_FLOAT:
			return "D32_FLOAT"sv;
		case DXGI_FORMAT_R32_FLOAT:
			return "R32_FLOAT"sv;
		case DXGI_FORMAT_R24G8_TYPELESS:
			return "R24G8_TYPELESS"sv;
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
			return "D24_UNORM_S8_UINT"sv;
		case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
			return "R24_UNORM_X8_TYPELESS"sv;
		case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
			return "X24_TYPELESS_G8_UINT"sv;
		case DXGI_FORMAT_R16_TYPELESS:
			return "R16_TYPELESS"sv;
		case DXGI_FORMAT_R16_FLOAT:
			return "R16_FLOAT"sv;
		case DXGI_FORMAT_D16_UNORM:
			return "D16_UNORM"sv;
		case DXGI_FORMAT_R16_UNORM:
			return "R16_UNORM"sv;
		case DXGI_FORMAT_R8_UNORM:
			return "R8_UNORM"sv;
		case DXGI_FORMAT_B8G8R8A8_UNORM:
			return "B8G8R8A8_UNORM"sv;
		default:
			return "?"sv;
		}
	}

	namespace
	{
		std::string BindFlagsToString(UINT a_flags)
		{
			std::string result;
			const auto  add = [&](UINT a_bit, std::string_view a_name) {
				if (a_flags & a_bit) {
					if (!result.empty()) {
						result += '|';
					}
					result += a_name;
				}
			};
			add(D3D11_BIND_SHADER_RESOURCE, "SRV");
			add(D3D11_BIND_RENDER_TARGET, "RTV");
			add(D3D11_BIND_DEPTH_STENCIL, "DSV");
			add(D3D11_BIND_UNORDERED_ACCESS, "UAV");
			return result.empty() ? "none" : result;
		}
	}

	std::string DescribeTexture(ID3D11Texture2D* a_texture)
	{
		if (!a_texture) {
			return "null";
		}

		D3D11_TEXTURE2D_DESC desc{};
		a_texture->GetDesc(&desc);
		return std::format(
			"{}x{} mips={} array={} fmt={}({}) samples={} bind={}",
			desc.Width, desc.Height, desc.MipLevels, desc.ArraySize,
			FormatName(desc.Format), static_cast<int>(desc.Format),
			desc.SampleDesc.Count, BindFlagsToString(desc.BindFlags));
	}

	std::string DescribeSRV(ID3D11ShaderResourceView* a_srv)
	{
		if (!a_srv) {
			return "null";
		}

		D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
		a_srv->GetDesc(&desc);
		std::string mips;
		if (desc.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D) {
			mips = std::format(" mostDetailed={} mipLevels={}", desc.Texture2D.MostDetailedMip, desc.Texture2D.MipLevels);
		}
		return std::format("srv fmt={}({}) dim={}{}", FormatName(desc.Format), static_cast<int>(desc.Format), static_cast<int>(desc.ViewDimension), mips);
	}
}
