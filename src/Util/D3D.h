#pragma once

namespace CBRO::Util
{
	[[nodiscard]] std::string_view FormatName(DXGI_FORMAT a_format) noexcept;

	// "1280x800 mips=1 array=1 fmt=R24G8_TYPELESS samples=1 bind=DSV|SRV", or "null".
	[[nodiscard]] std::string DescribeTexture(ID3D11Texture2D* a_texture);

	[[nodiscard]] std::string DescribeSRV(ID3D11ShaderResourceView* a_srv);

	[[nodiscard]] inline ID3D11DeviceContext* GetContext() noexcept
	{
		const auto data = RE::BSGraphics::RendererData::GetSingleton();
		return data ? reinterpret_cast<ID3D11DeviceContext*>(data->context) : nullptr;
	}

	[[nodiscard]] inline ID3D11Device* GetDevice() noexcept
	{
		const auto data = RE::BSGraphics::RendererData::GetSingleton();
		return data ? reinterpret_cast<ID3D11Device*>(data->device) : nullptr;
	}
}
