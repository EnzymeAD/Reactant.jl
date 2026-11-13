abstract type AbstractClient end

function Base.:(==)(a::AbstractClient, b::AbstractClient)
    return typeof(a) === typeof(b) && a.client == b.client
end

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
