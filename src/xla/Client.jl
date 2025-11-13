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

"""
    DEFAULT_DEVICE :: Ref{Int}

0-based index of default device to use, by default 0 (first available device).
"""
const DEFAULT_DEVICE = Ref{Int}(0)

function default_device(client::AbstractClient)
    return addressable_devices(client)[DEFAULT_DEVICE[] + 1]
end
