abstract type AbstractClient end

Base.:(==)(a::AbstractClient, b::AbstractClient) = a.client == b.client

function client end
function free_client end
function num_devices end
function num_addressable_devices end
function process_index end
function devices end
function addressable_devices end
function get_device end
function get_addressable_device end
function platform_name end

default_device(client::AbstractClient) = first(addressable_devices(client))
